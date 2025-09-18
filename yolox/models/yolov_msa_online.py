#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

"""
YOLOV++ Online MSA Head 네트워크 - 실시간 비디오 객체 탐지를 위한 온라인 처리 모듈

이 파일은 YOLOV++의 온라인 추론을 위한 특화된 Head 네트워크를 구현합니다.
v_plus_head.py의 전체 비디오 처리 방식과 달리, 실시간 스트리밍 환경에서
단일 프레임씩 순차적으로 처리하면서도 시간적 정보를 활용할 수 있도록 설계되었습니다.

주요 차이점 vs v_plus_head.py:
1. 온라인 처리: 전체 비디오 → 단일 프레임 + 참조 히스토리
2. 메모리 효율성: 고정된 참조 윈도우로 메모리 사용량 제한
3. 실시간성: 스트리밍 환경에서 낮은 지연시간 보장
4. 적응적 집합: 현재 프레임과 과거 프레임들 간의 시간적 융합

핵심 구성 요소:
1. YOLOXHead: 온라인 특화 헤드 클래스
2. MSA_yolov_online: 온라인 시간적 특징 집합 모듈
3. Reference Frame Management: 과거 프레임 정보 관리
4. Adaptive Aggregation: 현재-과거 프레임 간 적응적 융합
"""

import copy
import math
import time

import numpy
from loguru import logger

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from scipy.optimize import linear_sum_assignment
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
from yolox.models.post_trans import MSA_yolov_online
from torchvision.ops import roi_align
from yolox.models.post_process import postprocess_pure as postprocess


class YOLOXHead(nn.Module):
    """
    YOLOV++ Online MSA Head 클래스
    
    실시간 비디오 객체 탐지를 위한 온라인 특화 헤드 네트워크입니다.
    v_plus_head.py와 달리 단일 프레임 단위로 처리하면서도 과거 프레임 정보를 활용합니다.
    
    주요 특징:
    1. 온라인 처리: 실시간 스트리밍 환경 지원
    2. 참조 프레임 관리: 고정된 윈도우 크기로 메모리 효율성 확보
    3. MSA_yolov_online: 현재-과거 프레임 간 시간적 융합
    4. 단순화된 구조: 실시간 처리를 위한 최적화
    
    처리 과정:
    1. 기본 YOLOX 특징 추출 및 예측
    2. 고품질 proposal 선별 (RPN 역할)
    3. 과거 프레임과의 시간적 융합
    4. 최종 탐지 결과 생성
    """
    
    def __init__(
            self,
            num_classes,                   # 클래스 수 (ImageNet VID: 30)
            width=1.0,                     # 네트워크 폭 스케일링 인수
            strides=[8, 16, 32],           # FPN 각 레벨의 stride
            in_channels=[256, 512, 1024],  # 각 FPN 레벨의 입력 채널 수
            act="silu",                    # 활성화 함수 타입
            depthwise=False,               # Depthwise conv 사용 여부
            heads=4,                       # Multi-head attention의 head 수
            drop=0.0                       # Dropout 비율
    ):
        """
        YOLOV++ Online Head 초기화
        
        v_plus_head.py 대비 간소화된 파라미터:
        - 온라인 처리에 특화된 설정들만 유지
        - 복잡한 모드 설정들 제거로 단순화
        - 실시간 성능에 집중한 구조
        
        Args:
            num_classes: 탐지할 클래스 수 (ImageNet VID: 30)
            width: 네트워크 채널 수 스케일링 (기본 256 * width)
            strides: FPN 레벨별 stride [8, 16, 32]
            in_channels: FPN 레벨별 입력 채널 [256, 512, 1024]
            act: 활성화 함수 ("silu", "relu" 등)
            depthwise: Depthwise separable conv 사용 여부
            heads: MSA의 attention head 수 (기본 4)
            drop: Attention dropout 비율
        """
        super().__init__()

        # === 온라인 처리 파라미터 (v_plus_head.py와 동일한 기본값) ===
        self.Afternum = 30        # 각 프레임당 최종 proposal 수 (RPN 역할)
        self.Prenum = 750         # Pre-filtering proposal 수 (1차 선별)
        self.simN = 30            # 시간적 융합에 사용할 proposal 수 (Afternum과 동일)
        self.nms_thresh = 0.75    # Pre-NMS 임계값
        self.n_anchors = 1        # Anchor-free 방식이므로 1
        self.num_classes = num_classes      # 클래스 수 (ImageNet VID: 30)
        self.decode_in_inference = True     # 추론 시 디코딩 수행 (배포 시 False)

        # === 기본 YOLOX 구조의 레이어들 ===
        # v_plus_head.py와 동일한 기본 구조 유지
        self.cls_convs = nn.ModuleList()    # 분류 특징 추출 conv layers
        self.reg_convs = nn.ModuleList()    # 회귀 특징 추출 conv layers  
        self.cls_preds = nn.ModuleList()    # 분류 예측 heads
        self.reg_preds = nn.ModuleList()    # 회귀 예측 heads
        self.obj_preds = nn.ModuleList()    # Objectness 예측 heads
        self.cls_convs2 = nn.ModuleList()   # 온라인 시간적 특징 추출용 추가 레이어

        # === 온라인 시간적 특징 집합 모듈 ===
        self.width = int(256 * width)  # 특징 차원 (기본 256)
        
        # MSA_yolov_online: 온라인 환경에 특화된 시간적 특징 집합 모듈
        # - 현재 프레임과 참조 프레임들 간의 적응적 융합
        # - 고정된 참조 윈도우로 메모리 효율성 확보
        self.trans = MSA_yolov_online(
            dim=self.width,              # 입력 특징 차원 (256)
            out_dim=4 * self.width,      # 출력 특징 차원 (1024) - 4배 확장
            num_heads=heads,             # Multi-head attention head 수 (4)
            attn_drop=drop               # Attention dropout 비율
        )
        
        # === 공통 구조 레이어들 ===
        self.stems = nn.ModuleList()  # 각 FPN 레벨의 특징 차원 통일 레이어들
        
        # 최종 예측 헤드: MSA 후 확장된 특징(1024)에서 클래스 예측
        # 31 = num_classes(30) + 1(background/confidence)
        self.linear_pred = nn.Linear(int(4 * self.width), 31)

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_convs2.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, other_result=[], labels=None, imgs=None, nms_thresh=0.5):
        """
        YOLOV++ Online MSA Head Forward Pass
        
        실시간 비디오 객체 탐지를 위한 온라인 처리 함수입니다.
        v_plus_head.py와 달리 단일 프레임 + 참조 히스토리를 활용한 처리를 수행합니다.
        
        Args:
            xin: FPN 출력 리스트 [dark3, dark4, dark5] - 현재 프레임의 특징맵들
            other_result: 과거 프레임들의 MSA 결과 (온라인 시간적 융합용)
                        빈 리스트: 첫 프레임 또는 단일 프레임 처리
                        딕셔너리: 과거 프레임의 특징, 점수, 박스 정보
            labels: 훈련 시 Ground Truth 라벨 [batch, max_objects, 5(cls+xywh)]
            imgs: 원본 이미지 텐서 (크기 정보 등에 사용)
            nms_thresh: NMS 임계값 (추론 시 사용, 기본 0.5)
            
        Returns:
            훈련 시: get_losses 결과 (9개 손실 값)
            추론 시: (탐지 결과, MSA 메타 정보)
            
        온라인 처리 특징:
        1. 단일 프레임 기반 처리 (배치 크기 제한)
        2. 과거 프레임 정보를 other_result로 받아서 융합
        3. 메모리 효율적인 참조 윈도우 관리
        4. 실시간 성능 최적화
        """
        
        # === 출력 저장용 변수 초기화 ===
        outputs = []               # 훈련용 원시 출력들
        outputs_decode = []        # 디코딩된 출력들 (sigmoid 적용됨)
        origin_preds = []          # L1 손실용 원본 예측들
        x_shifts = []              # Grid X 좌표들
        y_shifts = []              # Grid Y 좌표들
        expanded_strides = []      # 각 앵커의 stride 값들
        before_nms_features = []   # 시간적 집합용 분류 특징들
        before_nms_regf = []       # 시간적 집합용 회귀 특징들

        # === 온라인 처리 파라미터 설정 ===
        # 동적으로 조정 가능하지만 일반적으로 고정값 사용
        self.Afternum = 30         # 각 프레임당 최종 proposal 수
        self.simN = 30             # 시간적 융합에 사용할 proposal 수

        # === STEP 1: 기본 특징 추출 및 예측 (v_plus_head.py와 동일) ===
        # 각 FPN 레벨(dark3, dark4, dark5)에서 특징 추출 및 기본 예측 수행
        for k, (cls_conv, cls_conv2, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.cls_convs2, self.reg_convs, self.strides, xin)
        ):
            # 1.1) Stem을 통해 통일된 차원으로 변환
            x = self.stems[k](x)  # [B, in_channels[k], H, W] -> [B, 256*width, H, W]
            
            # 1.2) 각 브랜치별 특징 추출
            reg_feat = reg_conv(x)    # 회귀 특징 [B, 256*width, H, W]
            cls_feat = cls_conv(x)    # 분류 특징 [B, 256*width, H, W]
            cls_feat2 = cls_conv2(x)  # 온라인 시간적 융합용 분류 특징 [B, 256*width, H, W]

            # 1.3) 기본 예측 헤드들을 통한 1차 예측 (YOLOX와 동일)
            obj_output = self.obj_preds[k](reg_feat)    # Objectness [B, 1, H, W]
            reg_output = self.reg_preds[k](reg_feat)    # Bounding box [B, 4, H, W]
            cls_output = self.cls_preds[k](cls_feat)    # Classification [B, num_classes, H, W]
            
            # 1.4) 훈련/추론에 따른 출력 처리
            if self.training:
                # 훈련 시: 원시 출력과 디코딩된 출력 모두 저장
                output = torch.cat([reg_output, obj_output, cls_output], 1)  # [B, 5+num_classes, H, W]
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1  # [B, 5+num_classes, H, W]
                )
                
                # 그리드 좌표 생성 및 앵커 위치 계산
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()  # output: [B, H*W, 5+num_classes]
                )
                x_shifts.append(grid[:, :, 0])  # X 좌표 [1, H*W]
                y_shifts.append(grid[:, :, 1])  # Y 좌표 [1, H*W]
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])  # [1, H*W]
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                
                # L1 손실을 위한 원본 예측 저장
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                outputs.append(output)  # 훈련용 원시 출력 저장
                before_nms_features.append(cls_feat2)  # 시간적 융합용 분류 특징
                before_nms_regf.append(reg_feat)       # 시간적 융합용 회귀 특징
            else:
                # 추론 시: 디코딩된 출력만 필요
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

                # 시간적 융합용 특징들 저장 (추론 시에도 필요)
                before_nms_features.append(cls_feat2)  # 온라인 융합용 분류 특징
                before_nms_regf.append(reg_feat)       # 온라인 융합용 회귀 특징
            
            outputs_decode.append(output_decode)  # 디코딩된 출력 저장

        # === STEP 2: 기본 후처리 및 특징 준비 ===
        outputs_decode = outputs_decode
        before_nms_regf = before_nms_regf
        before_nms_features = before_nms_features
        self.hw = [x.shape[-2:] for x in outputs_decode]  # 각 FPN 레벨의 H, W 저장

        # 2.1) 다중 스케일 출력을 하나로 합치고 디코딩
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2
                                   ).permute(0, 2, 1)  # [B, total_anchors, 5+num_classes]
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type())  # 앵커 좌표 디코딩

        # 2.2) 고품질 Proposal 선별 (RPN 역할) - v_plus_head.py의 postprocess_widx와 유사
        pred_result, pred_idx = self.postpro_woclass(
            decode_res,               # [B, total_anchors, 5+num_classes]
            num_classes=30,          # 클래스 수
            nms_thre=self.nms_thresh, # Pre-NMS 임계값 (0.75)
            topK=self.Afternum       # 최종 선별할 proposal 수 (30)
        )
        # pred_result: List[Tensor] - 각 프레임별 선별된 detection [N_selected, 7+num_classes]
        # pred_idx: List[Tensor] - 각 프레임별 선별된 인덱스들 [N_selected]

        # === STEP 3: 온라인 시간적 융합 준비 ===
        res_dict = {}  # 메타 정보 저장용 딕셔너리
        
        # 3.1) 특징을 평면화하여 후속 처리 준비
        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        ).permute(0, 2, 1)  # [B, total_anchors, 256*width] - 분류 특징
        reg_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_regf], dim=2
        ).permute(0, 2, 1)   # [B, total_anchors, 256*width] - 회귀 특징

        # 3.2) 선별된 proposal들의 특징과 점수 추출 (v_plus_head.py의 find_feature_score와 동일)
        features_cls, features_reg, cls_scores, fg_scores, boxes = self.find_feature_score(
            cls_feat_flatten, pred_idx,  # 평면화된 특징, 선별된 인덱스
            reg_feat_flatten, imgs,      # 회귀 특징, 원본 이미지
            pred_result                  # 1차 예측 결과
        )
        
        # 3.3) 배치 차원 추가 (온라인 MSA 모듈 입력 형태에 맞춤)
        features_reg = features_reg.unsqueeze(0)   # [1, N, 256] - 배치 차원 추가
        features_cls = features_cls.unsqueeze(0)   # [1, N, 256] - 배치 차원 추가
        
        # 3.4) 추론 시 데이터 타입 통일
        if not self.training:
            cls_scores = cls_scores.to(cls_feat_flatten.dtype)
            fg_scores = fg_scores.to(cls_feat_flatten.dtype)
            boxes = boxes.to(cls_feat_flatten.dtype)

        # 3.5) 현재 프레임 정보를 결과 딕셔너리에 저장
        res_dict['cls_feature'] = features_cls    # 현재 프레임 분류 특징
        res_dict['reg_feature'] = features_reg    # 현재 프레임 회귀 특징
        res_dict['cls_scores'] = cls_scores       # 현재 프레임 분류 점수
        res_dict['reg_scores'] = fg_scores        # 현재 프레임 foreground 점수
        res_dict['boxes'] = boxes                 # 현재 프레임 박스 좌표
        res_dict['msa'] = None                    # MSA 결과 (추후 업데이트)

        # === STEP 4: 온라인 처리 모드 분기 ===
        
        # 4.1) 첫 프레임 또는 단일 프레임 처리
        if not self.training and other_result == [] and cls_feat_flatten.shape[0] == 1:
            # 참조할 과거 프레임이 없는 경우: 기본 YOLOX 결과만 반환
            return self.postprocess_single_img(pred_result, self.num_classes), res_dict

        # 4.2) 과거 프레임과의 시간적 융합 (핵심!)
        if not self.training and other_result != []:
            # 현재 프레임과 과거 프레임(other_result) 정보를 결합
            features_cls = torch.cat([features_cls, other_result['cls_feature'].unsqueeze(0)], dim=1)
            features_reg = torch.cat([features_reg, other_result['reg_feature'].unsqueeze(0)], dim=1)
            cls_scores = torch.cat([cls_scores, other_result['cls_scores']])
            fg_scores = torch.cat([fg_scores, other_result['reg_scores']])

        # === STEP 5: 온라인 MSA 시간적 융합 수행 ===
        trans_cls, msa = self.trans(
            features_cls, features_reg,  # 현재+과거 프레임 특징들 [1, N_total, 256]
            cls_scores, fg_scores,       # 현재+과거 프레임 점수들
            other_result=other_result,   # 과거 프레임 메타 정보
            boxes=boxes,                 # 박스 좌표 정보
            simN=self.simN               # 시간적 융합에 사용할 proposal 수 (30)
        )
        # trans_cls: MSA 후 융합된 분류 특징 [N_current, 1024]
        # msa: MSA attention weight 정보

        # === STEP 6: 최종 분류 예측 ===
        fc_output = self.linear_pred(trans_cls[:self.simN, :])  # [N_current, 31] -> [N_current, 30]
        res_dict['msa'] = msa[:self.simN, :]  # 현재 프레임의 MSA 정보 저장

        # 출력 형태 조정: [1, N_current, num_classes] 형태로 reshape 후 마지막 차원 제거
        fc_output = torch.reshape(fc_output, [1, -1, self.num_classes + 1])[:, :, :-1]

        # === STEP 7: 최종 결과 반환 ===
        if self.training:
            # === 훈련 시: 손실 계산 ===
            # 기본 YOLOX 손실 + 온라인 MSA의 refined 손실 계산
            return self.get_losses(
                imgs,                      # 원본 이미지들
                x_shifts,                  # Grid X 좌표들
                y_shifts,                  # Grid Y 좌표들
                expanded_strides,          # 각 앵커의 stride 값들
                labels,                    # GT 라벨들
                torch.cat(outputs, 1),     # 기본 YOLOX 예측들 결합
                origin_preds,              # L1 손실용 원본 예측들
                dtype=xin[0].dtype,        # 데이터 타입
                refined_cls=fc_output,     # MSA 후 개선된 분류 예측
                idx=pred_idx,              # 선별된 proposal 인덱스들
                pred_res=pred_result,      # 1차 예측 결과들
            )
        else:
            # === 추론 시: 최종 후처리 ===
            
            # 7.1) 최고 신뢰도 클래스 추출
            class_conf, class_pred = torch.max(fc_output, -1, keepdim=False)
            
            # 7.2) 온라인 MSA 결과로 최종 후처리 수행
            # v_plus_head.py의 postprocess와 동일하지만 단일 프레임 처리에 최적화
            result = postprocess(
                copy.deepcopy(pred_result[:1]),  # 현재 프레임의 기본 예측 (첫 번째만)
                self.num_classes,                # 클래스 수 (30)
                fc_output,                       # MSA 후 개선된 분류 예측
                nms_thre=nms_thresh              # NMS 임계값
            )

            # 7.3) 최종 결과 반환
            return result, res_dict
            # result: NMS 등 후처리가 완료된 최종 탐지 결과
            # res_dict: 다음 프레임 처리를 위한 메타 정보 (특징, 점수, MSA 등)

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        훈련 시 앵커 기반 예측을 절대 좌표로 변환하는 함수
        
        v_plus_head.py의 동일한 함수와 완전히 같은 기능을 수행합니다.
        Grid cell 상대 좌표를 이미지 절대 좌표로 변환합니다.
        
        Args:
            output: 예측 출력 [B, 5+classes, H, W]
            k: FPN 레벨 인덱스 (0=stride8, 1=stride16, 2=stride32)
            stride: 해당 레벨의 stride 값
            dtype: 텐서 데이터 타입
            
        Returns:
            output: 변환된 예측 [B, H*W, 5+classes]
            grid: 그리드 좌표 [1, H*W, 2]
            
        변환 공식:
        - x, y: (예측값 + grid_x/y) * stride  # 중심점 좌표
        - w, h: exp(예측값) * stride          # 폭/높이 (항상 양수가 되도록 exp 적용)
        """
        grid = self.grids[k]  # 캐시된 그리드 가져오기

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes  # [x, y, w, h, obj, cls1, cls2, ...] 총 5+classes 채널
        hsize, wsize = output.shape[-2:]  # 특징맵 크기
        
        # 그리드 크기가 변경되었으면 새로 생성
        if grid.shape[2:4] != output.shape[2:4]:
            # meshgrid로 각 grid cell의 좌표 생성 (0 ~ H-1, 0 ~ W-1)
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # [x, y] 순서로 스택하여 [1, 1, H, W, 2] 형태로 만듦
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid  # 캐시에 저장

        # 출력을 [B, 1, channels, H, W] -> [B, 1, H, W, channels] -> [B, H*W, channels]로 reshape
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)  # [1, H*W, 2]로 평면화
        
        # 좌표 변환: grid cell 상대 좌표 -> 이미지 절대 좌표
        output[..., :2] = (output[..., :2] + grid) * stride  # 중심점 (x, y)
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # 크기 (w, h)
        
        return output, grid

    def decode_outputs(self, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None):
        """
        선별된 proposal들의 특징과 점수를 추출하는 함수
        
        v_plus_head.py의 동일한 함수와 같은 기능을 수행하지만,
        온라인 처리를 위해 self.simN(30)개로 제한합니다.
        
        Args:
            features: 분류 특징맵들 [batch, total_features, channels]
            idxs: 각 프레임별 선별된 proposal 인덱스들 [batch] of indices
            reg_features: 회귀 특징맵들 [batch, total_features, channels] 
            imgs: 원본 이미지들 (크기 정보용, 현재 미사용)
            predictions: 1차 예측 결과들 [batch] of [N, 5+classes]
            roi_features: ROI 특징들 (현재 미사용)
            
        Returns:
            features_cls: 선별된 분류 특징들 [total_selected, channels]
            features_reg: 선별된 회귀 특징들 [total_selected, channels]
            cls_scores: 분류 신뢰도 점수들 [total_selected]
            fg_scores: Foreground 점수들 [total_selected] 
            boxes: 박스 위치들 [total_selected, 4]
            
        온라인 최적화:
        - 각 프레임당 최대 self.simN(30)개 proposal만 선택
        - 메모리 효율성을 위한 고정 크기 제한
        - 실시간 처리 성능 최적화
        """
        features_cls = []  # 선별된 분류 특징들
        features_reg = []  # 선별된 회귀 특징들
        cls_scores = []    # 분류 신뢰도 점수들
        fg_scores = []     # Foreground 점수들
        boxes = []         # 박스 위치들
        
        # 각 프레임별로 선별된 proposal들의 정보 추출
        for i, feature in enumerate(features):
            # 온라인 처리를 위해 최대 simN(30)개로 제한
            features_cls.append(feature[idxs[i][:self.simN]])              # 분류 특징 [≤30, channels]
            features_reg.append(reg_features[i, idxs[i][:self.simN]])      # 회귀 특징 [≤30, channels]
            cls_scores.append(predictions[i][:self.simN, 5])              # 최고 클래스 점수 [≤30]
            fg_scores.append(predictions[i][:self.simN, 4])               # Objectness 점수 [≤30]
            boxes.append(predictions[i][:self.simN, :4])                  # 박스 좌표 [≤30, 4]
            
        # 모든 프레임의 선별된 정보들을 하나로 합치기
        features_cls = torch.cat(features_cls)  # [total_selected, channels]
        features_reg = torch.cat(features_reg)  # [total_selected, channels]
        cls_scores = torch.cat(cls_scores)      # [total_selected]
        fg_scores = torch.cat(fg_scores)        # [total_selected]
        boxes = torch.cat(boxes)                # [total_selected, 4]
        
        return features_cls, features_reg, cls_scores, fg_scores, boxes

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            refined_cls,
            idx,
            pred_res,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        ref_targets = []
        num_fg = 0.0
        num_gts = 0.0
        ref_masks = []
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                ref_target[:, -1] = 1

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # cls_target = F.one_hot(
                #     gt_matched_classes.to(torch.int64), self.num_classes
                # ) * pred_ious_this_matching.unsqueeze(-1)
                cls_target_onehot = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                )
                cls_target = cls_target_onehot * pred_ious_this_matching.unsqueeze(-1)
                fg_idx = torch.where(fg_mask)[0]
                # print(num_gt)
                # print(fg_idx)
                # print(idx[batch_idx])
                # print(pred_ious_this_matching)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                fg = 0

                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_res[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                for ele_idx, ele in enumerate(idx[batch_idx]):
                    loc = torch.where(fg_idx == ele)[0]

                    if len(loc):
                        ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                        fg += 1
                        continue
                    if max_iou.values[ele_idx] >= 0.6:
                        max_idx = int(max_iou.indices[ele_idx])
                        # print(max_iou.values[ele_idx])
                        # print(torch.max(cls_target[max_idx,:]))
                        # TODO  values do not match
                        ref_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                        fg += 1
                    # elif max_iou.values[ele_idx]<0.3:
                    #     ref_target[ele_idx, :] = 0#1 - max_iou.values[ele_idx]
                    else:
                        ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]

                # print('num_gt:',num_gt,"fg_pred:",fg)

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            ref_targets.append(ref_target[:, :self.num_classes])
            ref_masks.append(ref_target[:, -1] == 0)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        ref_targets = torch.cat(ref_targets, 0)

        fg_masks = torch.cat(fg_masks, 0)
        ref_masks = torch.cat(ref_masks, 0)
        # print(sum(ref_masks)/ref_masks.shape[0])
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        # TODO
        # before_nms_features = torch.cat(
        #     [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        # ).permute(0, 2, 1)
        # before_nms_features_flatten = torch.flatten(before_nms_features,start_dim=0,end_dim=1)
        # trans_features = before_nms_features_flatten[fg_masks].unsqueeze(0)
        # cls_trans_pred = self.trans_preds[0].forward(trans_features)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        loss_ref = (
                       self.bcewithlog_loss(
                           refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]
                       )
                   ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 3.0

        loss = reg_weight * loss_iou + loss_obj + 2 * loss_ref + loss_l1 + loss_cls
        # if loss_obj > 20:
        #     loss = loss_l1+loss_ref
        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            2 * loss_ref,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 4.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def postpro_woclass(self, prediction, num_classes, nms_thre=0.75, topK=75, features=None):
        """
        온라인 처리를 위한 고품질 Proposal 선별 함수 (클래스 정보 제외)
        
        v_plus_head.py의 postprocess_widx와 유사하지만 온라인 처리에 최적화된 버전입니다.
        RPN(Region Proposal Network) 역할을 하여 시간적 융합에 사용할 고품질 proposal들을 선별합니다.
        
        Args:
            prediction: 기본 예측 결과들 [batch, total_anchors, 5+classes]
            num_classes: 클래스 수 (30)
            nms_thre: NMS 임계값 (기본 0.75)
            topK: 최종 선별할 proposal 수 (기본 75, self.Afternum=30으로 오버라이드됨)
            features: 특징맵들 (현재 미사용)
            
        Returns:
            output: 선별된 예측들 [batch] of [N_selected, 7+classes]
                   형태: [x1,y1,x2,y2,obj_conf,cls_conf,cls_pred,all_cls_scores]
            output_index: 선별된 인덱스들 [batch] of indices [N_selected]
            
        선별 과정:
        1. 좌표 변환: center 형태 -> corner 형태 (x1,y1,x2,y2)
        2. Objectness 기반 상위 Prenum(750)개 선별
        3. Pre-NMS 적용: 중복 제거
        4. 최종 topK(30)개 proposal 선택
        
        온라인 최적화:
        - 클래스 정보 기반 필터링 제외 (속도 향상)
        - 고정된 proposal 수로 메모리 효율성 확보
        - 실시간 처리를 위한 간소화된 파이프라인
        """
        self.topK = topK  # 최종 선별할 proposal 수 설정
        
        # === STEP 1: 좌표 형식 변환 (Center → Corner) ===
        # YOLOX 출력은 (center_x, center_y, width, height) 형태
        # NMS와 후처리를 위해 (x1, y1, x2, y2) corner 형태로 변환
        box_corner = prediction.new(prediction.shape)  # 동일한 크기의 새 텐서 생성
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1 = center_x - width/2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1 = center_y - height/2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2 = center_x + width/2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2 = center_y + height/2
        prediction[:, :, :4] = box_corner[:, :, :4]  # 원본에 corner 좌표 덮어쓰기
        
        # === STEP 2: 결과 저장용 변수 초기화 ===
        output = [None for _ in range(len(prediction))]           # 각 이미지별 선별된 예측들
        output_index = [None for _ in range(len(prediction))]     # 각 이미지별 선별된 앵커 인덱스들
        features_list = []  # 특징 리스트 (현재 미사용)
        
        # === STEP 3: 배치별 Proposal 선별 수행 ===
        for i, image_pred in enumerate(prediction):
            # 빈 예측인 경우 건너뛰기
            if not image_pred.size(0):
                continue
                
            # === STEP 3-1: 클래스 신뢰도 계산 및 Detection 구성 ===
            # 가장 높은 클래스 신뢰도와 해당 클래스 추출
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            # Detection 형태로 재구성: [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred, all_cls_scores]
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)

            # === STEP 3-2: Objectness 기반 상위 선별 ===
            # 온라인 처리 최적화: 클래스 정보를 고려하지 않고 objectness만으로 1차 선별
            conf_score = image_pred[:, 4]  # Objectness 점수만 사용 (class_conf 곱하지 않음)
            top_pre = torch.topk(conf_score, k=self.Prenum)  # 상위 Prenum(750)개 선별
            sort_idx = top_pre.indices[:self.Prenum]  # 선별된 인덱스들
            detections_temp = detections[sort_idx, :]  # 선별된 detection들
            
            # === STEP 3-3: Pre-NMS 적용 ===
            # 중복 제거를 위한 NMS 적용
            nms_out_index = torchvision.ops.batched_nms(
                detections_temp[:, :4],                    # bbox 좌표
                detections_temp[:, 4] * detections_temp[:, 5],  # combined confidence
                detections_temp[:, 6],                     # 클래스 라벨
                nms_thre,                                  # NMS 임계값
            )

            # === STEP 3-4: 최종 TopK 선별 ===
            # NMS 통과한 것들 중에서 최종 topK개만 선별
            topk_idx = sort_idx[nms_out_index[:self.topK]]  # 원본 인덱스로 변환
            output[i] = detections[topk_idx, :]  # 최종 선별된 detection들
            output_index[i] = topk_idx           # 최종 선별된 인덱스들

        return output, output_index

    def find_similar_round2(self, features, sort_results):

        key_feature = features[0]
        support_feature = features[0]
        sort_res = sort_results
        most_sim_feature = support_feature  # [sort_res.indices[:, :support_pro], :]
        softmax_value = sort_res

        # softmax_value = torch.unsqueeze(softmax_value, dim=-1).repeat([1, 1, int(most_sim_feature.shape[-1])])
        soft_sim_feature = softmax_value @ support_feature  # torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = soft_sim_feature  # torch.cat([soft_sim_feature, key_feature], dim=-1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)

        return cls_feature

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):
        """
        단일 프레임 후처리 함수 (첫 프레임 또는 참조 프레임이 없는 경우)
        
        과거 프레임 정보가 없어서 시간적 융합을 수행할 수 없는 경우
        기본 YOLOX의 탐지 결과만을 사용하여 최종 후처리를 수행합니다.
        
        Args:
            prediction: 기본 예측 결과들 [batch] of [N, 7+classes]
            num_classes: 클래스 수 (30)
            conf_thre: 신뢰도 임계값 (기본 0.001)
            nms_thre: NMS 임계값 (기본 0.5)
            
        Returns:
            output_ori: 후처리된 탐지 결과들 [batch] of [N_final, 7]
                       형태: [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred]
            
        처리 과정:
        1. 신뢰도 필터링: obj_conf * cls_conf >= threshold
        2. NMS 적용: 중복 탐지 제거
        3. 최종 탐지 결과 반환
        
        사용 시점:
        - 첫 프레임 (과거 참조 없음)
        - 단일 프레임 처리 모드
        - 온라인 시스템 초기화 시
        """
        output_ori = [None for _ in range(len(prediction))]  # 각 이미지별 최종 결과
        prediction_ori = copy.deepcopy(prediction)  # 원본 예측 복사
        
        # 각 이미지별로 개별 후처리 수행
        for i, detections in enumerate(prediction):
            # 빈 예측인 경우 건너뛰기
            if not detections.size(0):
                continue

            detections_ori = prediction_ori[i]  # 현재 이미지의 예측들

            # === STEP 1: 신뢰도 기반 필터링 ===
            # Combined confidence: objectness * class_confidence >= threshold
            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]  # 조건을 만족하는 detection만 유지
            
            # === STEP 2: NMS 적용 ===
            # 중복 탐지 제거를 위한 Non-Maximum Suppression
            nms_out_index = torchvision.ops.batched_nms(
                detections_ori[:, :4],                    # bbox 좌표
                detections_ori[:, 4] * detections_ori[:, 5],  # combined confidence
                detections_ori[:, 6],                     # 클래스 라벨
                nms_thre,                                 # NMS 임계값
            )
            
            # === STEP 3: 최종 결과 저장 ===
            detections_ori = detections_ori[nms_out_index]  # NMS 통과한 detection만 유지
            output_ori[i] = detections_ori  # 최종 결과 저장
            
        return output_ori


"""
=== YOLOV++ Online MSA Head 네트워크 전체 요약 ===

이 파일은 YOLOV++의 실시간 비디오 객체 탐지를 위한 온라인 처리 특화 Head 네트워크입니다.

## 주요 특징 vs v_plus_head.py

### 1. 온라인 처리 최적화
- **전체 비디오 → 단일 프레임**: 실시간 스트리밍 환경에 최적화
- **참조 윈도우**: 고정된 과거 프레임 수로 메모리 사용량 제한
- **순차 처리**: 프레임별 순차적 처리로 낮은 지연시간 보장

### 2. 간소화된 구조
- **복잡한 모드 제거**: v_plus_head.py의 다양한 설정 옵션들 간소화
- **핵심 기능 집중**: MSA 시간적 융합에만 집중
- **실시간 성능**: 속도와 정확도의 균형점 추구

## 핵심 구성 요소

### 1. YOLOXHead 클래스
```python
# 온라인 특화 파라미터
Afternum = 30        # 각 프레임당 최종 proposal 수
Prenum = 750         # Pre-filtering proposal 수  
simN = 30            # 시간적 융합에 사용할 proposal 수
nms_thresh = 0.75    # Pre-NMS 임계값
```

### 2. MSA_yolov_online 모듈
- **온라인 시간적 융합**: 현재 + 과거 프레임 간 적응적 융합
- **메모리 효율성**: 고정된 참조 윈도우로 메모리 사용량 제한
- **실시간 성능**: 단일 프레임 단위 처리 최적화

## Forward Pass 7단계

### STEP 1: 기본 특징 추출 및 예측
- 각 FPN 레벨에서 기본 YOLOX 예측 수행
- 온라인 시간적 융합용 특징 추출 (cls_convs2)

### STEP 2: 기본 후처리 및 특징 준비
- 다중 스케일 출력 합치기 및 디코딩
- postpro_woclass를 통한 고품질 proposal 선별

### STEP 3: 온라인 시간적 융합 준비
- 선별된 proposal들의 특징과 점수 추출
- 현재 프레임 정보를 res_dict에 저장

### STEP 4: 온라인 처리 모드 분기
- 첫 프레임: 기본 YOLOX 결과만 반환
- 이후 프레임: 과거 프레임과 융합

### STEP 5: 온라인 MSA 시간적 융합 (핵심!)
- 현재 + 과거 프레임 특징들을 MSA로 융합
- 시간적 정보를 활용한 향상된 분류 특징 생성

### STEP 6: 최종 분류 예측
- MSA 후 융합된 특징으로 최종 분류 수행
- linear_pred를 통한 클래스 예측

### STEP 7: 최종 결과 반환
- 훈련 시: 손실 계산
- 추론 시: 후처리 후 탐지 결과 + 메타 정보 반환

## 주요 함수들

### 1. postpro_woclass
- **역할**: RPN 역할의 고품질 proposal 선별
- **특징**: 클래스 정보 제외로 속도 향상
- **출력**: 최대 30개 proposal per frame

### 2. find_feature_score  
- **역할**: 선별된 proposal들의 특징과 점수 추출
- **최적화**: simN(30)개로 제한하여 메모리 효율성 확보

### 3. postprocess_single_img
- **역할**: 첫 프레임 또는 단일 프레임 후처리
- **사용 시점**: 과거 참조 프레임이 없는 경우

## 온라인 처리 메커니즘

### 1. 참조 프레임 관리
```python
# 현재 프레임 정보 저장
res_dict = {
    'cls_feature': features_cls,    # 분류 특징
    'reg_feature': features_reg,    # 회귀 특징  
    'cls_scores': cls_scores,       # 분류 점수
    'reg_scores': fg_scores,        # Foreground 점수
    'boxes': boxes,                 # 박스 좌표
    'msa': msa_results             # MSA 정보
}
```

### 2. 시간적 융합 과정
```python
# 과거 프레임과 결합
if other_result != []:
    features_cls = torch.cat([current, past], dim=1)
    
# MSA 융합 수행  
trans_cls, msa = self.trans(features_cls, ...)
```

## 성능 특징

### 1. 실시간 성능
- **단일 프레임 처리**: 배치 크기 1로 제한
- **고정 proposal 수**: 메모리 사용량 예측 가능
- **간소화된 파이프라인**: 불필요한 계산 제거

### 2. 메모리 효율성
- **참조 윈도우**: 과거 프레임 수 제한
- **Proposal 제한**: 프레임당 최대 30개
- **온디맨드 처리**: 필요시에만 MSA 수행

### 3. 정확도 유지
- **시간적 정보 활용**: 과거 프레임과의 융합
- **MSA 어텐션**: 관련성 높은 정보 선별적 활용
- **적응적 융합**: 프레임 간 유사도 기반 가중치

## v_plus_head.py와의 주요 차이점

| 특징 | v_plus_head.py | yolov_msa_online.py |
|------|----------------|---------------------|
| 처리 방식 | 전체 비디오 배치 | 단일 프레임 + 참조 |
| 메모리 사용 | 가변적 (비디오 길이에 따라) | 고정적 (참조 윈도우) |
| 지연시간 | 높음 (전체 비디오 필요) | 낮음 (실시간 처리) |
| 설정 복잡도 | 높음 (다양한 모드) | 낮음 (핵심 기능만) |
| 사용 환경 | 오프라인 처리 | 온라인 스트리밍 |

## 실제 사용 시나리오

### 1. 실시간 비디오 스트리밍
- 웹캠, CCTV 등의 실시간 영상 처리
- 자율주행차의 실시간 객체 탐지
- 로봇비전 시스템

### 2. 온라인 추론 파이프라인
```python
# 첫 프레임
result1, meta1 = model(frame1, other_result=[])

# 이후 프레임들  
result2, meta2 = model(frame2, other_result=meta1)
result3, meta3 = model(frame3, other_result=meta2)
...
```

이 구조를 통해 YOLOV++는 실시간 환경에서도 시간적 정보를 활용한 
고성능 비디오 객체 탐지를 실현할 수 있습니다.
"""
