#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

"""
YOLOV++ Head 네트워크 - 비디오 객체 탐지를 위한 시간적 특징 집합 모듈

이 파일은 YOLOV++의 핵심인 Head 네트워크를 구현합니다.
기존 YOLOX Head에 시간적 특징 집합 기능을 추가하여 비디오 객체 탐지 성능을 향상시킵니다.

주요 구성 요소:
1. YOLOVHead: 메인 헤드 클래스
   - 기본 YOLOX 구조 + 비디오 특화 레이어들
   - MSA (Multi-Scale Attention) 또는 LocalAggregation 선택 가능
   - OTA 기반 라벨 할당 및 개선된 손실 함수

2. Forward Pass 6단계:
   STEP 1: 기본 특징 추출 및 예측
   STEP 2: 기본 후처리 및 Proposal 선별
   STEP 3: 시간적 특징 집합 준비
   STEP 4: 특징 집합 방식 선택 (MSA/LocalAgg)
   STEP 5: 손실 계산 (훈련 시)
   STEP 6: 추론 시 최종 후처리

3. 핵심 혁신:
   - 시간적 특징 집합: 다중 프레임 정보 활용
   - 분리된 회귀 브랜치: 분류/회귀 별도 처리
   - 개선된 손실: refined cls/iou/obj 손실 추가
   - 적응적 proposal 선별: 고품질 detection만 집합
"""

import copy
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger

# YOLOV++ 특화 모듈들
from yolox.models.post_process import postprocess, get_linking_mat  # 후처리 및 연결 매트릭스
from yolox.models.post_trans import MSA_yolov, LocalAggregation, visual_attention  # 시간적 특징 집합 모듈들
from yolox.utils import bboxes_iou  # IoU 계산
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou  # 박스 변환 및 GIoU
from .losses import IOUloss  # IoU 손실
from .network_blocks import BaseConv, DWConv  # 기본 conv 블록들
from yolox.utils.debug_vis import visual_predictions  # 디버깅용 시각화
from matplotlib import pyplot as plt  # 플롯팅
class YOLOVHead(nn.Module):
    """
    YOLOV++ Head 네트워크 클래스
    
    이 클래스는 YOLOV++의 핵심 혁신인 시간적 특징 집합을 담당합니다.
    기존 YOLOX Head에 비디오 특화 기능들을 추가한 구조입니다.
    
    주요 특징:
    1. 다중 프레임 특징 집합 (MSA 또는 LocalAggregation)
    2. 분리된 분류/회귀 브랜치 (decouple_reg)
    3. 개선된 재신뢰도 계산 (reconf)
    4. OTA 기반 라벨 할당
    5. 비디오 특화 추가 conv 레이어들
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
            drop=0.0,                      # Dropout 비율
            use_score=True,                # 신뢰도 점수 사용 여부
            defualt_p=30,                  # 각 프레임당 proposal 수 (Afternum)
            sim_thresh=0.75,               # 프레임 간 유사도 임계값
            pre_nms=0.75,                  # Pre-NMS 임계값
            ave=True,                      # Average pooling 사용 여부
            defulat_pre=750,               # Pre-filtering proposal 수 (Prenum)
            test_conf=0.001,               # 테스트 시 신뢰도 임계값
            use_mask=False,                # Attention mask 사용 여부
            gmode=True,                    # Global mode (다중 프레임 처리)
            lmode=False,                   # Local mode (연속 프레임 처리) - 실제로는 lframe=0
            both_mode=False,               # 둘 다 사용
            localBlocks=1,                 # Local aggregation 블록 수
            **kwargs                       # 추가 설정들 (agg_type, reconf, decouple_reg 등)
    ):
        """
        YOLOV++ Head 초기화
        
        핵심 파라미터 설명:
        - defualt_p (Afternum): 각 프레임당 최종 proposal 수. 논문에서는 30 사용
        - sim_thresh: 프레임 간 유사도 임계값. MSA에서 중요한 하이퍼파라미터
        - heads: Multi-head attention의 head 수. 보통 4 사용
        - gmode: Global 모드로 전체 비디오에서 샘플링된 프레임들 활용
        - agg_type: 'msa' (Multi-Scale Attention) 또는 'localagg' (Local Aggregation)
        - reconf: 재신뢰도 계산 여부. reconf를 true로 설정하면, MSA를 통해 cls뿐만 아니라 reg의 obj_pred(objectness)도 계산된다. 
                  params 는 늘어나지만 성능이 더 좋아지는 효과
        - decouple_reg: 분류와 회귀를 위한 별도 MSA 사용. decouple_reg를 true로 설정하면, MSA를 통해 cls와 reg를 별도로 계산한다. 
                        또한 reconf는 자동으로 true로 설정된다. 이 방법이 논문에 나오는 방법.
        - ota_mode: OTA 라벨 할당 방식 사용 여부
        - vid_cls: 비디오 분류 특징 추출 레이어 사용 여부 (기본 True)
        - vid_reg: 비디오 회귀 특징 추출 레이어 사용 여부 (기본 False)
        - localBlocks: local aggregation시 transformer 블록 수. 기본적으로 1로 설정하여 online inference시 속도를 높인다.

        참고 설명:
        - width: 네트워크 폭 스케일링 인수로서 모델의 크기를 조절. 기본 채널 수인 256에 width값을 곱해서 실제 사용할 채널 수를 결정한다.
        - use_score: 신뢰도 점수 사용 여부. 기본 True. 이게 있으면 attention 계산 시 cls_sccore(분류 신뢰도)와 fg_score(foreground 신뢰도)를 사용한다.
                     만약 false로 설정하면 신뢰도 점수를 무시하고 순수한 특징 유사도만으로 attention 계산을 한다. 하지만 이는 유사성 문제를 일으킬 수 있다.
        - default_pre: 이 값은 가장 처음 YOLOX가 가장 처음 detection을 할 때 상위 default_pre개의 proposal을 선택하는 기준이 된다.
        - pre_nms: 이 값은 가장 처음 YOLOX가 가장 처음 detection을 할 때 pre_nms를 이용해서 중복된 proposal을 제거하는 기준이 된다.
        """
        super().__init__()

        # === 기본 파라미터 설정 ===
        self.Afternum = defualt_p        # 각 프레임당 최종 proposal 수 (기본 30)
        self.Prenum = defulat_pre        # Pre-filtering proposal 수 (기본 750)
        self.simN = defualt_p            # 유사도 계산용 proposal 수
        self.nms_thresh = pre_nms        # Pre-NMS 임계값 (0.75)
        self.n_anchors = 1               # Anchor-free 방식이므로 1
        self.use_score = use_score       # 신뢰도 점수 사용 여부
        self.num_classes = num_classes   # 클래스 수 (ImageNet VID: 30)
        self.decode_in_inference = True  # 추론 시 디코딩 수행 (배포 시 False로 설정)
        
        # === 모드 설정 ===
        self.gmode = gmode      # Global 모드: 전체 비디오에서 샘플링된 프레임들 활용
        self.lmode = lmode      # Local 모드: 연속된 프레임들 활용 (실제로는 lframe=0)
        self.both_mode = both_mode  # 두 모드 결합 사용

        # === 기본 YOLOX 구조의 레이어들 ===
        # 각 FPN 레벨마다 분류/회귀 브랜치를 별도로 구성
        self.cls_convs = nn.ModuleList()  # 분류 특징 추출 conv layers (2개의 3x3 conv)
        self.reg_convs = nn.ModuleList()  # 회귀 특징 추출 conv layers (2개의 3x3 conv)
        self.cls_preds = nn.ModuleList()  # 분류 예측 heads (1x1 conv)
        self.reg_preds = nn.ModuleList()  # 회귀 예측 heads (1x1 conv)
        self.obj_preds = nn.ModuleList()  # Objectness 예측 heads (1x1 conv)
        
        # === 비디오 특화 추가 레이어들 (YOLOV++ 핵심 혁신) ===
        if kwargs.get('vid_cls', True):
            self.cls_convs2 = nn.ModuleList()  # 비디오 분류 특징 추출용 추가 conv layers
        if kwargs.get('vid_reg', False):
            self.reg_convs2 = nn.ModuleList()  # 비디오 회귀 특징 추출용 추가 conv layers

        # === 특징 집합 관련 파라미터 ===
        self.width = int(256 * width)    # 특징 차원 (보통 256)
        self.sim_thresh = sim_thresh     # 프레임 간 유사도 임계값 (0.75)
        self.ave = ave                   # Average pooling 사용 여부
        self.use_mask = use_mask         # Attention mask 사용 여부

        # === 시간적 특징 집합 모듈 선택 (YOLOV++의 핵심 혁신) ===
        # 어떤 방식으로 프레임 간 특징을 집합할지 결정하는 중요한 부분
        if kwargs.get('ota_mode', False):  # OTA 라벨 할당 사용 시
            if kwargs.get('agg_type', 'localagg') == 'localagg':
                # === Local Aggregation 방식 ===
                # 지역적 시간 정보를 활용한 특징 집합
                self.agg = LocalAggregation(
                    dim=self.width,           # 특징 차원 (256)
                    heads=heads,              # Multi-head 수 (4)
                    attn_drop=drop,           # Dropout 비율
                    blocks=localBlocks,       # Local 블록 수 (1)
                    **kwargs
                )
                # 간단한 Linear 예측 헤드들 (특징 차원 그대로 사용)
                self.cls_pred = nn.Linear(self.width, num_classes)  # 분류 예측
                self.obj_pred = nn.Linear(self.width, 1)            # Objectness 예측
                self.reg_pred = nn.Linear(self.width, 4)            # 회귀 예측 (x,y,w,h)
                
            elif kwargs.get('agg_type', 'localagg') == 'msa':
                # === Multi-Scale Attention (MSA) 방식 - YOLOV++의 주요 혁신 ===
                # 전역적 프레임 간 유사도를 계산하여 적응적 특징 융합
                self.agg = MSA_yolov(
                    dim=self.width,              # 입력 특징 차원 (256)
                    out_dim=4 * self.width,      # 출력 특징 차원 (1024) - 4배 확장
                    num_heads=heads,             # Multi-head attention head 수 (4)
                    attn_drop=drop,              # Attention dropout
                    reconf=kwargs.get('reconf', False)  # 재신뢰도 계산 여부
                )
                
                # === 분리된 회귀 브랜치 (Decoupled Regression) ===
                # 분류와 회귀를 위한 별도의 MSA 모듈 사용
                if kwargs.get('decouple_reg', False):
                    self.agg_iou = MSA_yolov(
                        dim=self.width,
                        out_dim=4 * self.width,
                        num_heads=heads,
                        attn_drop=drop,
                        reconf=True  # 회귀 브랜치는 항상 reconf 사용
                    )
                
                # MSA 후 확장된 차원에서 예측
                self.cls_pred = nn.Linear(4 * self.width, num_classes)  # 1024 -> 30
                if kwargs.get('reconf', False):
                    self.obj_pred = nn.Linear(4 * self.width, 1)        # 1024 -> 1
                self.cls_convs2 = nn.ModuleList()
                """
                추가 설명:
                - agg_type = localagg로 설정하는 경우 online상황에서 reference window를 사용하여 aggregation하는 방법이다. 
                - 한편, MSA를 선택하면 reference window를 사용하지 않고 모든 프레임을 사용하여 aggregation하는 방법이다. 
                - regression box같은 경우는 YOLOX에서 나온 것을 그대로 사용하며, 
                    classification이나 objectness는 MSA를 통해 나온 feature를 이용하기 때문에 MSA를 선택하는 경우,
                    cls와 obj를 예측하기 위한 pred를 따로 정의해주는 모습이이다.
                """
                
        else:  # OTA 미사용 시
            if kwargs.get('reconf', False):
                # Reconf 사용 시 LocalAggregation
                self.agg = LocalAggregation(
                    dim=self.width, heads=heads, attn_drop=drop, 
                    blocks=localBlocks, **kwargs
                )
                self.cls_pred = nn.Linear(self.width, num_classes)
                self.obj_pred = nn.Linear(self.width, 1)
            else:
                # 기본적으로 MSA 사용
                self.agg = MSA_yolov(
                    dim=self.width, out_dim=4 * self.width,
                    num_heads=heads, attn_drop=drop, 
                    reconf=kwargs.get('reconf', False)
                )
                self.cls_pred = nn.Linear(4 * self.width, num_classes)
            self.cls_convs2 = nn.ModuleList()

        if both_mode:
            self.g2l = nn.Linear(int(4 * self.width), self.width)
        self.stems = nn.ModuleList()
        self.kwargs = kwargs
        Conv = DWConv if depthwise else BaseConv

        # === 각 FPN 레벨마다 분류/회귀 브랜치를 별도로 구성 ===
        for i in range(len(in_channels)):
            # 각 FPN 레벨마다 stem layer를 추가하여, 특징 차원을 일치시킨다.
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            # cls branch로 3x3 conv를 2개 추가한다.
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
            # reg branch로 3x3 conv를 2개 추가한다.
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
            # cls pred layer를 추가한다. 이때는 FSM을 위해 1x1 conv를 사용해 실제 object가 있을 법한 위치를 뽑기 위함이다.
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # reg pred layer를 추가한다. 이는 regression box를 예측하기 위한 layer이다.
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4, # (x,y,w,h)
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # obj pred layer를 추가한다. 이는 objectness를 예측하기 위한 layer이다.
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            # vid_cls, vid_reg가 True인 경우, 비디오 특화 특징 추출 레이어를 추가한다.
            if kwargs.get('vid_cls',True):
                # VOD cls branch로서 3x3 conv를 2개 추가한다.
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
            # VOD reg branch로서 3x3 conv를 2개 추가한다. 기본값은 False이며, 논문에서는 언급되지 않았다.
            if kwargs.get('vid_reg',False):
                self.reg_convs2.append(
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

        # === 손실 함수 및 기타 설정 ===
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none") # L1 손실 함수
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none") # BCE 손실 함수
        self.iou_loss = IOUloss(reduction="none") # IoU 손실 함수
        self.strides = strides # FPN 각 레벨의 stride
        self.ota_mode = kwargs.get('ota_mode',False) # OTA 라벨 할당 사용 여부
        self.grids = [torch.zeros(1)] * len(in_channels) # 각 FPN 레벨마다 그리드 좌표를 저장하기 위한 리스트

    def initialize_biases(self, prior_prob):
        """
        예측 헤드들의 bias를 초기화하는 함수
        
        분류와 objectness 예측의 초기 bias를 설정하여 훈련 초기에 안정적인 예측을 가능하게 합니다.
        prior_prob에 따라 sigmoid 출력이 해당 확률값을 가지도록 bias를 설정합니다.
        
        Args:
            prior_prob (float): 초기 확률값. 보통 0.01 사용으로 99% 배경, 1% 객체로 시작
                              이는 대부분의 앵커가 배경일 것이라는 사전 지식을 반영
        
        수학적 원리:
        sigmoid(x) = prior_prob 가 되도록 하는 x 값은
        x = log(prior_prob / (1 - prior_prob))
        따라서 bias = -log((1 - prior_prob) / prior_prob)
        """
        # 분류 예측 헤드들의 bias 초기화
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)  # [1, num_classes] 형태로 reshape
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))  # 로그 확률로 초기화
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # Objectness 예측 헤드들의 bias 초기화
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)  # [1, 1] 형태로 reshape
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))  # 동일한 로그 확률로 초기화
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5, lframe=0, gframe=32):
        """
        YOLOV++ Head Forward Pass - 6단계 처리 과정
        
        Args:
            xin: FPN 출력 리스트 [dark3, dark4, dark5] - 각각 다른 해상도의 특징맵
            labels: 훈련 시 Ground Truth 라벨 [batch, max_objects, 5(cls+xywh)]
            imgs: 원본 이미지 텐서 (크기 정보 등에 사용)
            nms_thresh: NMS 임계값 (추론 시 사용)
            lframe: 로컬 프레임 수 (연속된 프레임) - 실제로는 0
            gframe: 글로벌 프레임 수 (샘플링된 프레임) - 보통 32
            
        Returns:
            훈련 시: (total_loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg, 
                     loss_refined_cls, loss_refined_iou, loss_refined_obj)
            추론 시: (result, result_ori) - 후처리된 탐지 결과
        
        Forward Pass 6단계:
        1. 기본 특징 추출 및 예측 (YOLOX와 동일 + 비디오 특화 레이어)
        2. 기본 후처리 및 Proposal 선별 (NMS, OTA 등)
        3. 시간적 특징 집합 준비 (특징/점수 추출)
        4. 특징 집합 방식 선택 (MSA 또는 LocalAgg)
        5. 손실 계산 (기본 + refined 손실들)
        6. 추론 시 최종 후처리
        """
        
        # === STEP 1: 기본 특징 추출 및 예측 ===
        # 각 FPN 레벨에서 기본적인 분류/회귀/objectness 예측 수행
        outputs = []           # 훈련용 원시 출력들 [batch, anchors, 5+classes]
        outputs_decode = []    # 디코딩된 출력들 (sigmoid 적용됨)
        origin_preds = []      # L1 손실용 원본 예측들
        x_shifts = []          # Grid X 좌표들
        y_shifts = []          # Grid Y 좌표들  
        expanded_strides = []  # 각 앵커의 stride 값들
        raw_cls_features = []  # 시간적 집합용 분류 특징들
        raw_reg_features = []  # 시간적 집합용 회귀 특징들

        # 각 FPN 레벨 (dark3: stride=8, dark4: stride=16, dark5: stride=32)별로 처리
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # 1.1) Stem을 통해 통일된 차원으로 변환
            # 이때 H와 W는 각 level별 특징맵 그리드의 크기이다.
            x = self.stems[k](x)  # [B, in_channels[k], H, W] -> [B, 256*width, H, W]
            
            """
            Implementation Point:
            - frame들에 대해 reg용 frames들과 VOD cls용 frame들을 나누어서 처리한다.
            - Batch 차원에 대해 routing하여 frame들을 구분한다.

            suedo code:
            - reg_frames_idx = frame_routing(x)
            - reg_x = x[:reg_frames_idx, :, :, :]
            - vid_cls_x = x[reg_frames_idx:, :, :, :]
            - reg_feat = reg_conv(reg_x)
            - vid_feat = self.cls_convs2[k](vid_cls_x)
            """

            # 1.2) 기본 YOLOX 분류/회귀 브랜치를 통한 특징 추출
            reg_feat = reg_conv(x)  # 회귀 특징 [B, 256*width, H, W]
            cls_feat = cls_conv(x)  # 분류 특징 [B, 256*width, H, W]
            
            # 1.3) 비디오 특화 특징 추출 (YOLOV++ 추가 부분)
            if self.kwargs.get('vid_cls', True):
                vid_feat = self.cls_convs2[k](x)  # 비디오 분류 특징 [B, 256*width, H, W]
            if self.kwargs.get('vid_reg', False):
                vid_feat_reg = self.reg_convs2[k](x)  # 비디오 회귀 특징 [B, 256*width, H, W]

            # 1.4) 기본 예측 헤드들을 통한 1차 예측 (YOLOX와 동일)
            obj_output = self.obj_preds[k](reg_feat)    # Objectness [B, 1, H, W]
            reg_output = self.reg_preds[k](reg_feat)    # Bounding box [B, 4, H, W]
            cls_output = self.cls_preds[k](cls_feat)    # Classification [B, num_classes, H, W]
            
            # 1.5) 출력 저장 및 그리드 정보 생성 (훈련/추론에 따라 다름)
            if self.training:
                # 훈련 시: 원시 출력과 디코딩된 출력 모두 저장
                output = torch.cat([reg_output, obj_output, cls_output], 1)  # [B, 5+num_classes, H, W]
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1  # [B, 5+num_classes, H, W]
                )
                
                # 그리드 좌표 생성 및 앵커 위치 계산
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()  # output: [B, H*W, 5+num_classes], grid: [1, H*W, 2]
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
                    batch_size = reg_output.shape[0] # [B, 4, H, W] -> B
                    hsize, wsize = reg_output.shape[-2:] # [B, 4, H, W] -> H, W
                    reg_output = reg_output.view( # [B, 4, H, W] -> [B, n_anchors, 4, H, W]
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape( # [B, n_anchors, 4, H, W] -> [B, H*W, 4]
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())  # [B, H*W, 4]
                outputs.append(output)  # [B, H*W, 5+num_classes]
            else:
                # 추론 시: 디코딩된 출력만 필요
                output_decode = torch.cat( 
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1  # [B, 5+num_classes, H, W]
                )
            
            # 1.6) 시간적 특징 집합을 위한 특징 저장
            # 비디오 특화 특징이 있으면 우선 사용, 없으면 기본 특징 사용
            if self.kwargs.get('vid_cls', True):
                raw_cls_features.append(vid_feat)  # 비디오 분류 특징 사용 [B, 256*width, H, W]
            else:
                raw_cls_features.append(cls_feat) # 기본 분류 특징 사용 [B, 256*width, H, W]
            if self.kwargs.get('vid_reg', False):
                raw_reg_features.append(vid_feat_reg) # 비디오 회귀 특징 사용 [B, 256*width, H, W]
            else:
                raw_reg_features.append(reg_feat) # 기본 회귀 특징 사용 [B, 256*width, H, W]

            outputs_decode.append(output_decode)  # [B, 5+num_classes, H, W]
        
        # === STEP 2: 기본 후처리 및 Proposal 선별 ===
        # 2.1) 다중 스케일 출력을 하나로 합치고 디코딩
        self.hw = [x.shape[-2:] for x in outputs_decode]  # 각 레벨의 H, W 저장 [(H1,W1), (H2,W2), (H3,W3)]
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2
                                   ).permute(0, 2, 1)  # [B, total_anchors = H1*W1 + H2*W2 + H3*W3, 5+num_classes]
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type())  # 앵커 좌표 디코딩 [B, total_anchors, 5+num_classes]
        preds_per_frame = []  # 각 프레임별 proposal 수 저장용 List[int]

        # 2.2) 훈련 시 라벨 할당 (OTA Dynamic K Matching)
        if self.training:
            assigned_packs = self.get_fg_idx(imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                )
            ota_idxs, cls_targets, reg_targets,\
                obj_targets, fg_masks, num_fg, num_gts, l1_targets = assigned_packs
            # ota_idxs: List[Tensor] - 각 프레임별 선별된 인덱스들
            # cls_targets: [total_fg] - 분류 타겟
            # reg_targets: [total_fg, 4] - 회귀 타겟
            # obj_targets: [total_fg] - Objectness 타겟
            # fg_masks: [B, total_anchors] - Foreground 마스크, foreground 앵커들의 인덱스들만 True로 설정된 마스크
            # num_fg, num_gts: int - FG 수, GT 수
            # l1_targets: [total_fg, 4] - L1 손실 타겟

            if not self.ota_mode: ota_idxs = None  # OTA 미사용 시 None
        else:
            ota_idxs = None

        # 2.3) 특징을 평면화하여 후속 처리 준비
        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in raw_cls_features], dim=2  # 각각 [B, 256*width, H*W]
        ).permute(0, 2, 1)  # [B, total_anchors, 256*width]
        reg_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in raw_reg_features], dim=2   # 각각 [B, 256*width, H*W]
        ).permute(0, 2, 1)   # [B, total_anchors, 256*width]

        # 2.4) 1차 후처리로 고품질 proposal 선별
        # 이 단계에서 대부분의 낮은 품질 detection들이 걸러짐
        pred_result, agg_idx, refine_obj_masks, cls_label_reorder = self.postprocess_widx(
            decode_res,              # [B, total_anchors, 5+num_classes]
            num_classes=self.num_classes,
            nms_thre=self.nms_thresh,     # Pre-NMS 임계값 (0.75)
            ota_idxs=ota_idxs,            # List[Tensor] - OTA에서 선별된 인덱스들
        )
        # pred_result: List[Tensor] - 각 프레임별 선별된 detection [N_selected, 7+num_classes]
        # agg_idx: List[Tensor] - 각 프레임별 선별된 인덱스들 [N_selected]
        # refine_obj_masks: List[Tensor] - Objectness 마스크
        # cls_label_reorder: List[Tensor] - 재정렬된 클래스 라벨

        # 2.5) 각 프레임별 선별된 proposal 수 계산
        for p in agg_idx:  # agg_idx: List[Tensor or None]
            if p is None: 
                preds_per_frame.append(0)
            else: 
                preds_per_frame.append(p.shape[0])  # 각 프레임의 proposal 수 int

        # 2.6) 예외 처리: proposal이 없는 경우
        if sum(preds_per_frame) == 0 and self.training:
            # 훈련 시 proposal이 없으면 0 손실 반환
            return torch.tensor(0), 0, 0, 0, 0, 1, 0, 0, 0  # 9개 손실 값

        if not self.training and imgs.shape[0] == 1:
            # 추론 시 단일 프레임이면 1차 결과만 반환
            return pred_result, pred_result  # List[Tensor], List[Tensor]

        # === STEP 3: 시간적 특징 집합 준비 ===
        
        # 3.1) 선별된 proposal들의 특징과 점수 추출
        # 이 함수가 YOLOV++의 핵심: 고품질 proposal들만 선별하여 특징 집합 대상으로 삼음
        (features_cls, features_reg, cls_scores,
         fg_scores, locs, all_scores) = self.find_feature_score(
            cls_feat_flatten,    # 평면화된 분류 특징들
            agg_idx,            # 선별된 proposal 인덱스들
            reg_feat_flatten,   # 평면화된 회귀 특징들
            imgs,               # 원본 이미지
            pred_result         # 1차 예측 결과들
        )
        
        # 3.2) 빈 결과 체크 (추론 시에만)
        if features_cls == None and not self.training: 
            return pred_result, pred_result
        if features_cls.shape[0] == 0 and not self.training: 
            return pred_result, pred_result


        # 3.3) 특징을 배치 차원으로 확장하여 집합 모듈 입력 준비
        features_reg_raw = features_reg.unsqueeze(0)   # [1, N, C] - 배치 차원 추가
        features_cls_raw = features_cls.unsqueeze(0)   # [1, N, C] - 배치 차원 추가

        # 3.4) 데이터 타입 통일 및 위치 정보 준비
        cls_scores = cls_scores.to(cls_feat_flatten.dtype)  # 분류 신뢰도 점수
        fg_scores = fg_scores.to(cls_feat_flatten.dtype)    # Foreground 신뢰도 점수
        locs = locs.to(cls_feat_flatten.dtype)              # Bounding box 위치 정보
        locs = locs.view(1, -1, 4)  # [1, N, 4] - (x, y, w, h)

        # 3.5) 추가 인자들 준비 (특징 집합 모듈에 전달될 정보들)
        more_args = {
            'width': imgs.shape[-1],      # 이미지 폭
            'height': imgs.shape[-2],     # 이미지 높이
            'fg_score': fg_scores,        # Foreground 점수들
            'cls_score': cls_scores,      # 분류 점수들
            'all_scores': all_scores,     # 모든 클래스 점수들
            'lframe': lframe,             # 로컬 프레임 수 (0)
            'afternum': self.Afternum,    # proposal 수 (30)
            'gframe': gframe,             # 글로벌 프레임 수 (32)
            'use_score': self.use_score   # 점수 사용 여부
        }

        # === STEP 4: 특징 집합 방식 선택 (YOLOV++의 핵심 혁신!) ===
        if self.kwargs.get('agg_type', 'localagg') == 'localagg':
            # 4.1) Local Aggregation 방식
            # 지역적 시간 정보를 활용한 특징 집합
            features_cls, features_reg = self.agg(
                features_cls,    # 분류 특징들
                features_reg,    # 회귀 특징들  
                locs,           # 위치 정보
                **more_args     # 추가 인자들
            )

            # Local Aggregation 후 예측 (차원 변화 없음: 256 -> classes/1/4)
            cls_preds = self.cls_pred(features_cls)  # [N, num_classes]
            obj_preds = self.obj_pred(features_reg)  # [N, 1]
            if self.ota_mode:
                reg_preds = self.reg_pred(features_reg)  # [N, 4]
                reg_preds = torch.reshape(reg_preds, [-1, 4])
            else:
                reg_preds = None

            cls_preds = torch.reshape(cls_preds, [-1, self.num_classes])
            obj_preds = torch.reshape(obj_preds, [-1, 1])

        elif self.kwargs.get('agg_type', 'localagg') == 'msa':
            # 4.2) Multi-Scale Attention (MSA) 방식 - YOLOV++의 주요 혁신
            # 전역적 프레임 간 유사도를 계산하여 적응적 특징 융합
            kwargs = self.kwargs
            kwargs.update({'lframe': lframe, 'gframe': gframe, 'afternum': self.Afternum})

            """
            implementation point:
            - router를 달아서 바로 head로 갈 수 있도록 token-wise early exit branch를 생성한다.

            suedo code:
            - not_early_exit_idx, early_exit_idx = router(features_cls_raw, features_reg_raw)
            - features_cls, features_reg = self.agg(
                features_cls_raw[not_early_exit_idx],
                features_reg_raw[not_early_exit_idx],
                cls_scores[not_early_exit_idx],
                fg_scores[not_early_exit_idx],
                sim_thresh=self.sim_thresh,
                ave=self.ave,
                use_mask=self.use_mask,
                **kwargs
            )

            if self.kwargs.get('decouple_reg', False):
                _, features_reg = self.agg_iou(
                    features_cls_raw[not_early_exit_idx], 
                    features_reg_raw[not_early_exit_idx], 
                    cls_scores[not_early_exit_idx], 
                    fg_scores[not_early_exit_idx],
                    sim_thresh=self.sim_thresh,
                    ave=self.ave, use_mask=self.use_mask, **kwargs
                )
            else:
                features_reg = None

            - cls_preds = self.cls_pred(torch.cat([features_cls, features_cls_raw[early_exit_idx]], dim=0))  # [N, num_classes]
            - obj_preds = self.obj_pred(torch.cat([features_reg, features_reg_raw[early_exit_idx]], dim=0))  # [N, 1]
            - reg_preds = self.reg_pred(torch.cat([features_reg, features_reg_raw[early_exit_idx]], dim=0))  # [N, 4]
            """

            # MSA를 통한 시간적 특징 집합 (핵심!)
            features_cls, features_reg = self.agg(
                features_cls_raw,        # [1, N, 256] 분류 특징
                features_reg_raw,        # [1, N, 256] 회귀 특징
                cls_scores,              # 분류 신뢰도 점수들
                fg_scores,               # Foreground 신뢰도 점수들
                sim_thresh=self.sim_thresh,  # 유사도 임계값 (0.75)
                ave=self.ave,                # Average pooling 사용 여부
                use_mask=self.use_mask,      # Attention mask 사용 여부
                **kwargs
            )
            
            # 4.3) 분리된 회귀 브랜치 (Decoupled Regression)
            # 분류와 회귀를 위한 별도의 MSA 모듈 사용 - 성능 향상의 핵심
            if self.kwargs.get('decouple_reg', False):
                _, features_reg = self.agg_iou(
                    features_cls_raw, features_reg_raw, cls_scores, fg_scores,
                    sim_thresh=self.sim_thresh,
                    ave=self.ave, use_mask=self.use_mask, **kwargs
                )
            
            # 4.4) MSA 후 최종 예측 (확장된 차원에서: 1024 -> classes/1)
            cls_preds = self.cls_pred(features_cls)  # [N, num_classes]
            if self.kwargs.get('reconf', False):
                obj_preds = self.obj_pred(features_reg)  # [N, 1] - 재신뢰도 계산
                reg_preds = None  # reconf 모드에서는 reg 예측 안함
            else:
                obj_preds, reg_preds = None, None

        # === STEP 5: 손실 계산 (훈련 시) ===
        if self.training:
            # === STEP 5-1: 기본 출력 준비 ===
            outputs = torch.cat(outputs, 1)  # 모든 FPN 레벨의 출력 결합 [batch, total_anchors, 5+classes]
            
            # === STEP 5-2: Refined 타겟 생성 방식 선택 ===
            if not self.ota_mode:
                # === 5-2A: 비-OTA 모드 - IoU 기반 라벨 할당 ===
                # 전통적인 IoU 기반 방식으로 refined 타겟 생성
                stime = time.time()
                (refine_cls_targets,   # IoU 가중 분류 타겟들
                 refine_cls_masks,     # 분류 감독 마스크들 (애매한 IoU 제외)
                 refine_obj_targets,   # Objectness 타겟들
                 refine_obj_masks) = ( # Objectness 감독 마스크들
                    self.get_iou_based_label(pred_result,agg_idx,labels,outputs,reg_targets,cls_targets)
                )
                # 배치별 결과를 하나로 합치기
                refine_cls_targets = torch.cat(refine_cls_targets, 0)
                refine_cls_masks = torch.cat(refine_cls_masks, 0)
                refine_obj_targets = torch.cat(refine_obj_targets, 0)
                refine_obj_masks = torch.cat(refine_obj_masks, 0)
            else:
                # === 5-2B: OTA 모드 - 다양한 refined 타겟 생성 방식 ===
                # 기본적으로는 refined 타겟 없음 (OTA가 충분히 정확하므로)
                refine_cls_targets, refine_cls_masks, refine_obj_targets = None, None, None
                
                # === 5-2B-1: 정적 객체 탐지기 감독 사용 ===
                # 비디오 OTA를 사용하지 않을 때의 대안적 방식들
                if not self.kwargs.get('vid_ota',False):
                    
                    # === Case 1: CLS_OTA 모드 (기본값: True) ===
                    if self.kwargs.get('cls_ota',True):
                        # === Subcase 1-1: OTA FG와 분리 처리 ===
                        if not self.kwargs.get('cat_ota_fg',True):
                            # OTA foreground와 새로운 proposal을 분리하여 처리
                            refine_cls_targets = []
                            for i in range(len(ota_idxs)):
                                tmp_reorder = cls_label_reorder[i]  # 클래스 재정렬 정보
                                # 유효한 OTA 인덱스와 재정렬 정보가 있는 경우
                                if ota_idxs[i] != None and tmp_reorder!=None and len(tmp_reorder):
                                    # 재정렬된 순서로 분류 타겟 추출
                                    tmp_cls_targets = cls_targets[i][torch.stack(tmp_reorder)]
                                    refine_cls_targets.append(tmp_cls_targets)
                            
                            # 추출된 타겟들이 있으면 합치기, 없으면 빈 텐서 생성
                            if len(refine_cls_targets):
                                refine_cls_targets = torch.cat(refine_cls_targets, 0)
                            else:
                                refine_cls_targets = torch.cat(cls_targets, 0).new_zeros(0, self.num_classes)
                        
                        # === Subcase 1-2: OTA FG와 통합 처리 (기본값) ===
                        # 대부분의 경우 여기서 끝남 (refined 타겟 없이 OTA만 사용)
                    else:
                        # === Case 2: 비-CLS_OTA 모드 - IoU 기반 보조 사용 ===
                        (refine_cls_targets,
                         refine_cls_masks,     # IoU 기반 분류 마스크 생성
                         _,                    # refine_obj_targets는 사용 안함
                         iou_base_obj_masks) = (
                            self.get_iou_based_label(pred_result, agg_idx, labels, outputs, reg_targets, cls_targets)
                        )
                        refine_cls_targets = torch.cat(refine_cls_targets, 0)
                        refine_cls_masks = torch.cat(refine_cls_masks, 0)
                else:
                    # === 5-2B-2: 비디오 OTA 모드 - 고급 라벨 재할당 ===
                    # 시간적 집합 후 예측을 기반으로 라벨을 다시 할당하는 혁신적 방법
                    
                    # === Step 1: 비디오 예측 텐서 구성 ===
                    vid_preds = outputs.clone().detach()  # 기본 예측 복사
                    bidx_accum = 0  # 배치 누적 인덱스
                    
                    # 각 배치의 선별된 proposal들에 refined 예측값 할당
                    for b_idx,f_idx in enumerate(agg_idx):
                        if f_idx is None: continue  # 해당 배치에 선별된 proposal이 없으면 건너뛰기
                        
                        tmp_pred = vid_preds[b_idx,f_idx]  # 선별된 proposal들의 예측값
                        
                        # 기본 탐지기의 다른 예측들 억제 (매우 낮은 값으로 설정)
                        vid_preds[b_idx, :] = -1e3  # 선별되지 않은 앵커들은 무시
                        
                        # 시간적 집합으로 개선된 예측값들로 교체
                        tmp_pred[:,-self.num_classes:] = cls_preds[bidx_accum:bidx_accum+preds_per_frame[b_idx]]  # refined 분류
                        tmp_pred[:,4:5] = obj_preds[bidx_accum:bidx_accum+preds_per_frame[b_idx]]              # refined objectness
                        vid_preds[b_idx,f_idx] = tmp_pred  # 업데이트된 예측값 저장
                        
                        bidx_accum += preds_per_frame[b_idx]  # 다음 배치를 위한 인덱스 누적

                    # === Step 2: 비디오 예측 기반 OTA 라벨 할당 ===
                    # 개선된 예측을 사용하여 새로운 OTA 할당 수행
                    vid_packs = self.get_fg_idx(imgs,
                                                 x_shifts,
                                                 y_shifts,
                                                 expanded_strides,
                                                 labels,
                                                 vid_preds,      # 개선된 예측 사용
                                                 origin_preds,
                                                 dtype=xin[0].dtype,
                                                 )
                    # OTA 결과 언패킹
                    vid_fg_idxs, vid_cls_targets, vid_reg_targets, \
                        vid_obj_targets, vid_fg_masks, vid_num_fg, \
                        vid_num_gts, vid_l1_targets = vid_packs
                    
                    # === Step 3: 비디오 OTA 결과를 refined 타겟으로 변환 ===
                    refine_obj_masks,refine_cls_targets = [],[]
                    
                    for b_idx,f_idx in enumerate(agg_idx):
                        if f_idx is None: continue
                        f_idx = f_idx.cuda()  # GPU로 이동
                        
                        # 해당 배치의 선별된 proposal들 중 foreground 마스크 추출
                        refine_obj_masks.append(vid_fg_masks[b_idx][f_idx])
                        
                        # foreground proposal들의 분류 타겟 추출
                        tmp_cls_targets = []
                        for feature_idx in f_idx[vid_fg_masks[b_idx][f_idx]]:
                            # 비디오 OTA에서 매칭된 분류 타겟 찾기
                            cls_tar_idx = torch.where(feature_idx==vid_fg_idxs[b_idx])[0]
                            tmp_cls_targets.append(vid_cls_targets[b_idx][cls_tar_idx])
                        
                        # 분류 타겟들이 있으면 결합
                        if len(tmp_cls_targets):
                            tmp_cls_targets = torch.cat(tmp_cls_targets,0)
                            refine_cls_targets.append(tmp_cls_targets)
                    
                    # === Step 4: 최종 타겟 텐서 구성 ===
                    refine_obj_masks = torch.cat(refine_obj_masks,0)  # 객체 마스크 결합
                    if len(refine_cls_targets):
                        refine_cls_targets = torch.cat(refine_cls_targets, 0)  # 분류 타겟 결합
                    else:
                        # 분류 타겟이 없으면 빈 텐서 생성
                        refine_cls_targets = torch.cat(cls_targets, 0).new_zeros(0, self.num_classes)

            # === STEP 5-3: 기본 YOLOX 타겟들 결합 ===
            # 배치별로 나누어져 있던 기본 타겟들을 하나로 합치기
            cls_targets = torch.cat(cls_targets, 0)    # [total_fg, num_classes] - 분류 타겟
            reg_targets = torch.cat(reg_targets, 0)    # [total_fg, 4] - 회귀 타겟
            obj_targets = torch.cat(obj_targets, 0)    # [total_anchors, 1] - objectness 타겟
            fg_masks = torch.cat(fg_masks, 0)          # [total_anchors] - foreground 마스크
            if self.use_l1:
                l1_targets = torch.cat(l1_targets, 0)  # [total_fg, 4] - L1 손실 타겟

            # === STEP 5-4: 손실 계산 함수 호출 ===
            # 기본 YOLOX 손실 + YOLOV++ refined 손실들을 모두 계산
            return self.get_losses(
                outputs,              # 기본 YOLOX 예측들
                cls_targets,          # 기본 분류 타겟
                reg_targets,          # 기본 회귀 타겟
                obj_targets,          # 기본 objectness 타겟
                fg_masks,             # foreground 마스크
                num_fg,               # 총 foreground 수
                num_gts,              # 총 GT 수
                l1_targets,           # L1 손실 타겟
                origin_preds,         # L1 손실용 원본 예측
                cls_preds,            # refined 분류 예측
                obj_preds,            # refined objectness 예측
                reg_preds,            # refined 회귀 예측
                refine_obj_masks,     # refined objectness 마스크
                refine_cls_targets,   # refined 분류 타겟
                refine_cls_masks,     # refined 분류 마스크
                refine_obj_targets,   # refined objectness 타겟
            )
        else:
            # === STEP 6: 추론 시 최종 후처리 ===
            # 시간적 집합으로 개선된 예측들을 최종 탐지 결과로 변환
            
            # === STEP 6-1: 프레임별 예측 분할 ===
            # MSA로 집합된 예측들을 다시 각 프레임별로 분할
            cls_per_frame, obj_per_frame, reg_per_frame = [], [], []
            
            for i in range(len(preds_per_frame)):
                # === Refined Objectness 처리 ===
                if self.kwargs.get('reconf',False):
                    # 재신뢰도 계산이 활성화된 경우 refined objectness 사용
                    obj_per_frame.append(obj_preds[:preds_per_frame[i]].squeeze(-1))
                    obj_preds = obj_preds[preds_per_frame[i]:]  # 다음 프레임을 위해 슬라이싱
                
                # === Refined Classification 처리 ===
                # 시간적 집합으로 개선된 분류 예측 분할
                cls_per_frame.append(cls_preds[:preds_per_frame[i]])
                cls_preds = cls_preds[preds_per_frame[i]:]  # 다음 프레임을 위해 슬라이싱

            # === STEP 6-2: Objectness 소스 결정 ===
            # reconf가 비활성화된 경우 기본 objectness 사용
            if not self.kwargs.get('reconf',False): 
                obj_per_frame = None  # 기본 YOLOX objectness 사용

            # === STEP 6-3: 최종 후처리 수행 ===
            # NMS, 신뢰도 필터링 등의 표준 후처리 과정
            result, result_ori = postprocess(copy.deepcopy(pred_result),  # 기본 예측 결과 (백업용)
                                             self.num_classes,            # 클래스 수
                                             cls_per_frame,               # 프레임별 refined 분류 예측
                                             conf_output = obj_per_frame, # 프레임별 refined objectness (선택적)
                                             nms_thre = nms_thresh,       # NMS 임계값
                                             )
            # === STEP 6-4: 결과 반환 ===
            # result: NMS 등 후처리가 완료된 최종 탐지 결과
            # result_ori: 후처리 전 원본 결과 (디버깅/분석용)
            return result, result_ori

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        훈련 시 앵커 기반 예측을 절대 좌표로 변환하는 함수
        
        YOLOX는 anchor-free 방식이지만 내부적으로는 각 grid cell을 앵커로 취급합니다.
        이 함수는 grid cell 상대 좌표를 이미지 절대 좌표로 변환합니다.
        
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
        hsize, wsize = output.shape[-2:]  # 특징맵 크기, height, width
        
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
        """
        다중 스케일 예측 출력을 절대 좌표로 디코딩하는 함수
        
        여러 FPN 레벨의 출력이 이미 concatenate된 상태에서 한번에 디코딩을 수행합니다.
        각 레벨별로 다른 stride를 적용하여 올바른 스케일로 변환합니다.
        
        Args:
            outputs: 다중 스케일 예측 출력 [B, total_anchors, 5+classes]
                    total_anchors = H1*W1 + H2*W2 + H3*W3 (모든 FPN 레벨의 앵커 합)
            dtype: 텐서 데이터 타입
            flevel: 사용하지 않는 파라미터 (호환성 유지용)
            
        Returns:
            outputs: 디코딩된 예측 [B, total_anchors, 5+classes]
                    좌표가 이미지 절대 좌표계로 변환됨
                    
        처리 과정:
        1. 각 FPN 레벨별 그리드 좌표 생성
        2. 해당 레벨의 stride 값으로 채운 텐서 생성 
        3. 모든 레벨의 그리드와 stride를 concatenate
        4. 좌표 변환 수행
        """
        grids = []    # 각 레벨의 그리드 좌표들
        strides = []  # 각 레벨의 stride 값들
        
        # 각 FPN 레벨별로 그리드와 stride 정보 생성
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            # 해당 레벨의 그리드 좌표 생성 (0~H-1, 0~W-1)
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)  # [1, H*W, 2]
            grids.append(grid)
            
            # 해당 레벨의 모든 앵커에 동일한 stride 적용
            shape = grid.shape[:2]  # [1, H*W]
            strides.append(torch.full((*shape, 1), stride))  # [1, H*W, 1]

        # 모든 레벨의 그리드와 stride를 하나로 합치기
        grids = torch.cat(grids, dim=1).type(dtype)      # [1, total_anchors, 2]
        strides = torch.cat(strides, dim=1).type(dtype)  # [1, total_anchors, 1]

        # 좌표 디코딩: grid cell 상대좌표 -> 이미지 절대좌표
        outputs[..., :2] = (outputs[..., :2] + grids) * strides  # 중심점 좌표
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # 박스 크기
        
        return outputs
    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None):
        """
        선별된 proposal들의 특징과 점수를 추출하는 함수
        
        YOLOV++의 핵심: 1차 후처리에서 선별된 고품질 proposal들만을 대상으로
        시간적 특징 집합을 수행하기 위해 해당 proposal들의 특징과 점수를 추출합니다.
        
        Args:
            features: 분류 특징맵들 [batch, total_features, channels]
            idxs: 각 프레임별 선별된 proposal 인덱스들 [batch] of indices
            reg_features: 회귀 특징맵들 [batch, total_features, channels] 
            imgs: 원본 이미지들 (크기 정보용)
            predictions: 1차 예측 결과들 [batch] of [N, 5+classes]
            
        Returns:
            features_cls: 선별된 분류 특징들 [total_selected, channels]
            features_reg: 선별된 회귀 특징들 [total_selected, channels]
            cls_scores: 분류 신뢰도 점수들 [total_selected]
            fg_scores: Foreground 점수들 [total_selected] 
            locs: 박스 위치들 [total_selected, 4]
            all_scores: 모든 클래스 점수들 [total_selected, num_classes]
            
        이 함수의 중요성:
        - 메모리 효율성: 전체 특징 대신 선별된 것만 처리
        - 품질 보장: 고품질 proposal만 시간적 집합 대상
        - 속도 향상: 불필요한 계산 제거
        """
        features_cls = []  # 선별된 분류 특징들
        features_reg = []  # 선별된 회귀 특징들
        cls_scores, all_scores = [], []  # 점수들
        fg_scores = []     # Foreground 점수들
        locs = []          # 박스 위치들
        
        # 각 프레임별로 선별된 proposal들의 정보 추출
        for i, feature in enumerate(features):
            # 해당 프레임에 선별된 proposal이 없으면 스킵
            if idxs[i] is None or idxs[i] == []: 
                continue

            # 선별된 인덱스들의 특징과 점수 추출
            features_cls.append(feature[idxs[i]])              # 분류 특징 [N_selected, channels]
            features_reg.append(reg_features[i, idxs[i]])      # 회귀 특징 [N_selected, channels]
            cls_scores.append(predictions[i][:, 5])           # 최고 클래스 점수 [N_selected]
            fg_scores.append(predictions[i][:, 4])            # Objectness 점수 [N_selected]
            locs.append(predictions[i][:, :4])                # 박스 좌표 [N_selected, 4]
            all_scores.append(predictions[i][:, -self.num_classes:])  # 모든 클래스 점수 [N_selected, classes]
            
        # 선별된 proposal이 하나도 없는 경우
        if len(features_cls) == 0:
            return None, None, None, None, None, None
            
        # 모든 프레임의 선별된 정보들을 하나로 합치기
        features_cls = torch.cat(features_cls)  # [total_selected, channels]
        features_reg = torch.cat(features_reg)  # [total_selected, channels]
        cls_scores = torch.cat(cls_scores)      # [total_selected]
        fg_scores = torch.cat(fg_scores)        # [total_selected]
        locs = torch.cat(locs)                  # [total_selected, 4]
        all_scores = torch.cat(all_scores)      # [total_selected, classes]
        
        return features_cls, features_reg, cls_scores, fg_scores, locs, all_scores

    def get_losses(
            self,
            outputs,
            cls_targets,
            reg_targets,
            obj_targets,
            fg_masks,
            num_fg,
            num_gts,
            l1_targets,
            origin_preds,
            refined_cls,
            refined_obj,
            refined_reg,
            refined_obj_masks,
            refined_cls_targets,
            refined_cls_masks,
            refined_obj_targets,
    ):
        """
        YOLOV++ 전체 손실 계산 함수
        
        기본 YOLOX 손실과 YOLOV++ 개선 손실을 모두 계산하여 결합합니다.
        총 9개의 손실 항목을 계산하여 종합적인 학습 신호를 제공합니다.
        
        Args:
            outputs: 기본 YOLOX 예측 출력 [batch, total_anchors, 5+classes]
            cls_targets: 기본 분류 타겟 [total_fg, classes] (IoU 가중 one-hot)
            reg_targets: 기본 회귀 타겟 [total_fg, 4] (GT bbox 좌표)
            obj_targets: 기본 objectness 타겟 [total_anchors, 1] (binary)
            fg_masks: Foreground 마스크 [total_anchors] (boolean)
            num_fg: Foreground 앵커 수
            num_gts: GT 객체 수
            l1_targets: L1 손실용 타겟 [total_fg, 4]
            origin_preds: L1 손실용 원본 예측 [total_fg, 4]
            refined_cls: 시간적 집합 후 분류 예측 [total_selected, classes]
            refined_obj: 시간적 집합 후 objectness 예측 [total_selected, 1]
            refined_reg: 시간적 집합 후 회귀 예측 [total_selected, 4]
            refined_obj_masks: Refined objectness 마스크 [total_selected]
            refined_cls_targets: Refined 분류 타겟 [total_selected, classes]
            refined_cls_masks: Refined 분류 마스크 [total_selected] 
            refined_obj_targets: Refined objectness 타겟 [total_selected, 1]
            
        Returns:
            튜플 (9개 요소):
            - total_loss: 총 손실 (다른 모든 손실의 가중 합)
            - reg_weight * loss_iou: 가중치 적용된 IoU 손실
            - loss_obj: Objectness 손실 (BCE)
            - loss_cls: 분류 손실 (BCE)
            - loss_l1: L1 손실 (사용시)
            - num_fg / max(num_gts, 1): 정규화된 FG 비율
            - loss_refined_cls: 개선된 분류 손실
            - reg_weight * loss_refined_iou: 개선된 IoU 손실
            - loss_refined_obj: 개선된 objectness 손실
            
        손실 구성:
        1. 기본 YOLOX 손실 (4개):
           - IoU 손실: bounding box 회귀 정확도
           - Objectness 손실: 객체 존재 예측 정확도
           - 분류 손실: 클래스 분류 정확도
           - L1 손실: 좌표 회귀 매끄러운 학습
           
        2. YOLOV++ 개선 손실 (3개):
           - Refined 분류 손실: 시간적 집합 후 향상된 분류
           - Refined IoU 손실: 시간적 집합 후 향상된 회귀
           - Refined objectness 손실: 시간적 집합 후 향상된 신뢰도
           
        가중치 설정:
        - reg_weight = 3.0: 회귀 손실의 중요도 강조
        - 나머지 손실들은 동일 가중치 (1.0)
        - num_fg로 정규화하여 배치 크기 영향 제거
        """

        # 기본 YOLOX 예측들을 추출
        bbox_preds = outputs[:, :, :4]  # 박스 예측 [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # Objectness 예측 [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # 분류 예측 [batch, n_anchors_all, n_cls]

        # Refined objectness 타겟이 없으면 마스크를 기반으로 생성
        if refined_obj_targets == None:
            refined_obj_targets = refined_obj_masks.type_as(obj_targets)
            refined_obj_targets = refined_obj_targets.view(-1, 1)
            refined_obj_masks = refined_obj_masks.bool().squeeze(-1)

        # L1 손실 사용시 타겟 결합
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        # 정규화 인수 (제로 나누기 방지)
        num_fg = max(num_fg, 1)
        # === 기본 YOLOX 손실들 계산 ===
        # IoU 손실: Foreground 앵커들의 bounding box 회귀 정확도
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        
        # Objectness 손실: 모든 앵커의 객체 존재 예측 정확도
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        
        # 분류 손실: Foreground 앵커들의 클래스 분류 정확도 (IoU 가중 one-hot 대상)
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg



        # Refined IoU 손실 (현재 미사용)
        loss_refined_iou = 0

        # === YOLOV++ 개선 손실들 계산 ===
        if self.ota_mode:
            # OTA 모드에서의 refined 손실 계산
            if self.kwargs.get('reconf', True):
                # Refined objectness 손실: 시간적 집합 후 향상된 신뢰도 예측
                loss_refined_obj = (
                               self.bcewithlog_loss(refined_obj.view(-1, 1), refined_obj_targets)
                           ).sum() / num_fg
            else:
                loss_refined_obj = 0

            if self.kwargs.get('cls_ota',True):
                # OTA 방식의 refined 분류 손실: 모든 선별된 proposal에 대해 동일 가중치
                loss_refined_cls = (
                                       self.bcewithlog_loss(
                                           refined_cls.view(-1, self.num_classes)[refined_obj_masks], refined_cls_targets
                                       )
                                   ).sum() / num_fg
            else:
                # IoU 기반 방식의 refined 분류 손실: 유효한 마스크로 정규화
                refined_cls_fg = max(float(torch.sum(refined_cls_masks)), 1)
                loss_refined_cls = (
                                       self.bcewithlog_loss(
                                           refined_cls.view(-1, self.num_classes)[refined_cls_masks], refined_cls_targets[refined_cls_masks]
                                       )
                                   ).sum() / refined_cls_fg
        else:
            # 비-OTA 모드에서의 refined 손실 계산
            loss_refined_obj = 0
            
            # IoU 기반 refined 분류 손실
            refined_cls_fg = max(float(torch.sum(refined_cls_masks)), 1)
            loss_refined_cls = (
                self.bcewithlog_loss(
                    refined_cls.view(-1, self.num_classes)[refined_cls_masks], refined_cls_targets[refined_cls_masks]
                )
            ).sum() / refined_cls_fg
            
            # Reconf 사용시 refined objectness 손실 계산
            if self.kwargs.get('reconf',False):
                refined_obj_fg = max(float(torch.sum(refined_obj_masks)), 1)
                loss_refined_obj = (
                    self.bcewithlog_loss(
                        refined_obj.view(-1, 1)[refined_obj_masks], refined_obj_targets[refined_obj_masks]
                    )
                ).sum() / refined_obj_fg
        # L1 손실 계산 (사용시)
        if self.use_l1:
            # L1 손실: bounding box 회귀의 매끄러운 학습을 위한 추가 손실
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        # 회귀 손실 가중치 (박스 위치 정확도를 강조)
        reg_weight = 3.0

        # Refined objectness 손실 클리핑 (발산 방지)
        if loss_refined_obj > 15:
            # 손실이 너무 클 경우 15로 제한
            loss_refined_obj = loss_refined_obj / float(loss_refined_obj) * 15
            
        # === 총 손실 계산 ===
        # 기본 YOLOX 손실 + YOLOV++ 개선 손실
        loss = reg_weight * loss_iou + loss_obj + loss_l1 + loss_cls \
                + loss_refined_cls + reg_weight * loss_refined_iou + loss_refined_obj

        # 모든 손실 항목 반환 (9개)
        return (
            loss,                           # 1. 총 손실
            reg_weight * loss_iou,          # 2. 가중치 적용된 IoU 손실
            loss_obj,                       # 3. Objectness 손실
            loss_cls,                       # 4. 분류 손실
            loss_l1,                        # 5. L1 손실
            num_fg / max(num_gts, 1),       # 6. 정규화된 FG 비율
            loss_refined_cls,               # 7. 개선된 분류 손실
            reg_weight * loss_refined_iou,  # 8. 개선된 IoU 손실 (가중치 적용)
            loss_refined_obj,               # 9. 개선된 objectness 손실
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        """
        L1 손실을 위한 타겟 좌표 변환 함수
        
        GT 박스 좌표를 L1 손실 계산에 적합한 형태로 변환합니다.
        중심점 좌표는 grid cell 상대 좌표로, 크기는 로그 스케일로 변환합니다.
        
        Args:
            l1_target: 출력 타겟 텐서 [num_fg, 4] (수정될 대상)
            gt: GT 박스 좌표 [num_fg, 4] (x,y,w,h 절대좌표)
            stride: 해당 FPN 레벨의 stride
            x_shifts, y_shifts: 그리드 좌표들
            eps: 로그 계산시 수치 안정성을 위한 작은 값
            
        Returns:
            l1_target: 변환된 L1 타겟 [num_fg, 4]
            
        변환 공식:
        - x, y: (GT_center / stride) - grid_position  # grid cell 내 상대 위치
        - w, h: log((GT_size / stride) + eps)         # 로그 스케일 크기
        """
        # 중심점 좌표를 grid cell 상대 좌표로 변환
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts  # x 상대 위치
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts  # y 상대 위치
        
        # 크기를 로그 스케일로 변환 (양수 보장 및 수치 안정성)
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)  # log width
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)  # log height
        
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
        """
        OTA (Optimal Transport Assignment) 라벨 할당 알고리즘
        
        YOLOX/YOLOV++에서 사용하는 동적 라벨 할당 방식입니다.
        각 GT 객체에 대해 최적의 앵커들을 자동으로 선택하여 할당합니다.
        
        Args:
            batch_idx: 현재 배치 인덱스
            num_gt: 현재 이미지의 GT 객체 수
            total_num_anchors: 전체 앵커 수 (모든 FPN 레벨 합)
            gt_bboxes_per_image: GT 박스들 [num_gt, 4] (x,y,w,h)
            gt_classes: GT 클래스들 [num_gt]
            bboxes_preds_per_image: 예측 박스들 [total_anchors, 4]
            expanded_strides: 각 앵커의 stride [1, total_anchors]
            x_shifts, y_shifts: 그리드 좌표들 [1, total_anchors]
            cls_preds, bbox_preds, obj_preds: 예측 출력들
            labels: GT 라벨들
            imgs: 원본 이미지들
            mode: "gpu" 또는 "cpu" (메모리 부족시 CPU 모드 사용)
            
        Returns:
            gt_matched_classes: 매칭된 GT 클래스들 [num_fg]
            fg_mask: Foreground 마스크 [total_anchors]
            pred_ious_this_matching: 매칭된 IoU들 [num_fg]
            matched_gt_inds: 매칭된 GT 인덱스들 [num_fg]
            num_fg: Foreground 앵커 수
            
        OTA 알고리즘 과정:
        1. 기하학적 제약: GT 박스 내부 또는 중심 근처의 앵커만 고려
        2. 비용 계산: 분류 손실 + IoU 손실 + 기하학적 패널티
        3. 동적 K 매칭: 각 GT별로 최적의 K개 앵커 선택
        4. 충돌 해결: 여러 GT에 할당된 앵커는 가장 낮은 비용의 GT에 할당
        """

        # GPU 메모리 부족시 CPU 모드로 전환
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
        """
        OTA에서 기하학적 제약 조건을 확인하는 함수
        
        각 앵커가 GT 박스의 유효한 영역 내에 있는지 확인합니다.
        YOLOX는 두 가지 조건을 사용합니다:
        1. GT 박스 내부에 있는 앵커
        2. GT 박스 중심 근처 고정 반경 내의 앵커
        
        Args:
            gt_bboxes_per_image: GT 박스들 [num_gt, 4] (중심좌표 x,y,w,h)
            expanded_strides: 각 앵커의 stride [1, total_anchors]
            x_shifts, y_shifts: 그리드 좌표들 [1, total_anchors]
            total_num_anchors: 전체 앵커 수
            num_gt: GT 객체 수
            
        Returns:
            is_in_boxes_anchor: GT 박스 또는 중심 영역 내 앵커 마스크 [total_anchors]
            is_in_boxes_and_center: 두 조건 모두 만족하는 마스크 [num_gt, valid_anchors]
            
        기하학적 제약의 목적:
        - 관련 없는 앵커 제거로 계산량 감소
        - 공간적으로 합리적인 앵커만 후보로 고려
        - 중심 기반 제약으로 고품질 매칭 보장
        """
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
        """
        OTA의 핵심인 동적 K 매칭 알고리즘
        
        각 GT 객체별로 최적의 K개 앵커를 동적으로 선택하여 할당합니다.
        K값은 각 GT의 IoU 분포를 기반으로 자동으로 결정됩니다.
        
        Args:
            cost: 앵커-GT 간 비용 매트릭스 [num_gt, valid_anchors]
                 비용 = cls_loss + 3.0*iou_loss + geometric_penalty
            pair_wise_ious: 앵커-GT 간 IoU 매트릭스 [num_gt, valid_anchors]
            gt_classes: GT 클래스들 [num_gt]
            num_gt: GT 객체 수
            fg_mask: 유효한 앵커들의 마스크
            
        Returns:
            num_fg: 최종 선택된 foreground 앵커 수
            gt_matched_classes: 매칭된 GT 클래스들
            pred_ious_this_matching: 매칭에서의 IoU 값들
            matched_gt_inds: 매칭된 GT 인덱스들
            
        동적 K 매칭 과정:
        1. 각 GT별로 상위 10개 IoU의 합을 구해 K값 결정
        2. 각 GT마다 비용이 가장 낮은 K개 앵커 선택
        3. 여러 GT에 할당된 앵커는 가장 낮은 비용의 GT에만 할당
        4. 최종 매칭 결과 반환
        
        장점:
        - 적응적 K값으로 각 객체의 특성 반영
        - 고품질 매칭으로 훈련 안정성 향상
        - 중복 할당 방지로 명확한 학습 신호 제공
        """
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.kwargs.get('vid_dk',10), ious_in_boxes_matrix.size(1))
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

    def postprocess_widx(self, prediction, num_classes, nms_thre=0.5, ota_idxs=None,conf_thresh=0.001):
        """
        YOLOV++의 핵심: 시간적 집합을 위한 고품질 Proposal 선별 함수

        기본 YOLOX 예측에서 고품질 proposal들만 선별하여 시간적 특징 집합의 입력으로 사용합니다.
        
        Args:
            prediction: 기본 예측 결과들 [batch, total_anchors, 5+classes]
            num_classes: 클래스 수
            nms_thre: Pre-NMS 임계값 (기본 0.75)
            ota_idxs: OTA에서 선별된 인덱스들 (훈련시)
            conf_thresh: 신뢰도 임계값 (기본 0.001)
            
        Returns:
            output: 선별된 예측들 [batch] of [N_selected, 7+classes]
                   형태: [x1,y1,x2,y2,obj_conf,cls_conf,cls_pred,all_cls_scores]
            output_index: 선별된 인덱스들 [batch] of indices
            refined_obj_masks: 객체 마스크들 (refined 손실용)
            reorder_cls: 클래스 재정렬 정보 (OTA 매칭용)
            
        선별 과정:
        1. 좌표 변환: center 형태 -> corner 형태 (x1,y1,x2,y2)
        2. 신뢰도 필터링: obj_conf * cls_conf >= threshold
        3. Pre-NMS 적용: 중복 제거
        4. OTA 인덱스와의 결합 (훈련시)
        5. 최종 proposal 리스트 구성
        
        중요성:
        - 메모리 효율성: 전체 앵커 대신 선별된 proposal만 처리
        - 품질 보장: 고신뢰도 detection만 시간적 집합 대상
        - 속도 향상: 불필요한 계산 제거
        - 성능 향상: 노이즈가 적은 고품질 특징으로 집합 수행
        """
        # === STEP 1: RPN 역할의 TopK 예측 선별 ===
        # 전체 앵커 중에서 고품질 proposal들만 선별 (Region Proposal Network 역할)
        
        # === STEP 2: 좌표 형식 변환 (Center → Corner) ===
        # YOLOX 출력은 (center_x, center_y, width, height) 형태
        # NMS와 후처리를 위해 (x1, y1, x2, y2) corner 형태로 변환
        box_corner = prediction.new(prediction.shape)  # 동일한 크기의 새 텐서 생성
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1 = center_x - width/2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1 = center_y - height/2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2 = center_x + width/2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2 = center_y + height/2
        prediction[:, :, :4] = box_corner[:, :, :4]  # 원본에 corner 좌표 덮어쓰기

        # === STEP 3: 결과 저장용 변수 초기화 ===
        output = [None for _ in range(len(prediction))]           # 각 이미지별 선별된 예측들
        output_index = [None for _ in range(len(prediction))]     # 각 이미지별 선별된 앵커 인덱스들
        reorder_cls = [None for _ in range(len(prediction))]      # OTA 매칭용 클래스 재정렬 정보
        refined_obj_masks = []                                    # Refined 손실 계산용 객체 마스크들

        # === STEP 4: 배치별 Proposal 선별 수행 ===
        for i, image_pred in enumerate(prediction):
            # 현재 이미지의 refined objectness 마스크 초기화
            obj_mask = torch.zeros(0,1)  # [0, 1] - 시작은 빈 마스크

            # 빈 예측인 경우 건너뛰기
            if not image_pred.size(0):
                continue
                
            # === STEP 4-1: 클래스 신뢰도 계산 및 Detection 구성 ===
            # 가장 높은 클래스 신뢰도와 해당 클래스 추출
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            
            # Detection 형태로 재구성: [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred, all_cls_scores]
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)
            
            # === STEP 4-2: OTA Foreground 인덱스 처리 (훈련시) ===
            if ota_idxs is not None:
                if len(ota_idxs[i]) > 0 and self.kwargs.get('cat_ota_fg',True):
                    # OTA에서 선별된 foreground 앵커들을 우선적으로 포함
                    ota_idx = ota_idxs[i]  # 현재 이미지의 OTA foreground 인덱스들
                    output[i] = detections[ota_idx, :]  # OTA 인덱스에 해당하는 detection들
                    output_index[i] = ota_idx.cpu()     # CPU로 이동하여 저장
                    tmp_ota_mask = torch.ones_like(output_index[i]).unsqueeze(1)  # OTA는 모두 foreground
                    obj_mask = torch.cat((obj_mask, tmp_ota_mask))  # 객체 마스크에 추가

            # === STEP 4-3: 신뢰도 기반 필터링 ===
            # Combined confidence: objectness * class_confidence >= threshold
            conf_mask = (detections[:, 4] * detections[:, 5] >= conf_thresh).squeeze()
            
            # === STEP 4-4: Detection 수 제한 처리 ===
            minimal_limit = self.kwargs.get('minimal_limit',0)  # 최소 detection 수 제한
            maximal_limit = self.kwargs.get('maximal_limit',0)  # 최대 detection 수 제한
            
            # 최소 제한: 신뢰도가 낮아도 최소 개수는 보장
            if minimal_limit != 0:
                if conf_mask.sum() < minimal_limit:
                    # 신뢰도 순으로 상위 minimal_limit개 강제 선택
                    _, top_idx = torch.topk(detections[:, 4] * detections[:, 5], minimal_limit)
                    conf_mask[top_idx] = True
            
            # 최대 제한: 너무 많은 detection 방지 (메모리 효율성)
            if maximal_limit != 0:
                if conf_mask.sum() > maximal_limit:
                    logger.warning('current obj above conf thresh: %d' % conf_mask.sum())
                    
                    # 상위 점수 기반 선별: 가장 신뢰도 높은 maximal_limit개만 선택
                    _, top_idx = torch.topk(detections[:, 4] * detections[:, 5], maximal_limit)
                    conf_mask = torch.zeros_like(conf_mask)  # 기존 마스크 초기화
                    conf_mask[top_idx] = True  # 상위 인덱스들만 True로 설정

                    # 참고: NMS 기반 제한 방식도 고려했지만 현재는 단순 점수 기반 방식 사용
                    # 이유: 계산 효율성과 명확한 제어 가능

            # === STEP 4-5: 신뢰도 필터링 적용 ===
            conf_idx = torch.where(conf_mask)[0]  # 신뢰도 조건을 만족하는 앵커 인덱스들
            detections = detections[conf_mask]    # 조건을 만족하는 detection들만 선별
            
            # 선별된 detection이 없으면 다음 이미지로
            if not detections.size(0):
                refined_obj_masks.append(obj_mask)
                continue

            # === STEP 4-6: Pre-NMS 적용 (선택적) ===
            if self.kwargs.get('use_pre_nms',True):
                # 중복 제거를 위한 NMS 적용
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],                    # bbox 좌표
                    detections[:, 4] * detections[:, 5],  # combined confidence
                    detections[:, 6],                     # 클래스 라벨
                    nms_thre,                             # NMS 임계값
                )
            else:
                # NMS 사용하지 않으면 모든 detection 유지
                nms_out_index = torch.arange(detections.shape[0])

            # === STEP 4-7: OTA 인덱스와의 통합 처리 ===
            if ota_idxs!= None and ota_idxs[i] is not None and len(ota_idxs[i]) > 0:  # 훈련 모드
                if not self.kwargs.get('use_pre_nms',True):
                    if not self.kwargs.get('cat_ota_fg',True):
                        # OTA 인덱스와 일치하는 것만 foreground로 마킹
                        obj_mask = torch.zeros_like(nms_out_index).unsqueeze(1)
                        tmp_reorder = []  # OTA 매칭을 위한 재정렬 정보
                        
                        for j in nms_out_index:
                            # 현재 인덱스가 OTA foreground에 포함되는지 확인
                            tmp_idx = torch.where(ota_idxs[i]==conf_idx[j])[0]
                            if len(tmp_idx):
                                obj_mask[j] = 1  # foreground로 마킹
                                tmp_reorder.append(tmp_idx[0])  # 재정렬 정보 저장
                                
                        reorder_cls[i] = tmp_reorder  # 클래스 재정렬 정보 저장
                        abs_idx = conf_idx[nms_out_index].cpu()  # 절대 인덱스
                    else:
                        # OTA 인덱스와 새로운 detection들을 분리하여 처리
                        # OTA에 없는 새로운 detection들은 background로 처리
                        abs_idx_out_ota = torch.tensor([conf_idx[j] for j in nms_out_index if conf_idx[j] not in ota_idxs[i]])
                        abs_idx = abs_idx_out_ota
                        bg_mask = torch.zeros_like(abs_idx_out_ota).cpu()  # 새로운 것들은 background
                        obj_mask = torch.cat((obj_mask.type_as(bg_mask), bg_mask.unsqueeze(1)))
                else:
                    abs_idx = None  # Pre-NMS 사용시에는 별도 인덱스 불필요
            else:
                # 추론 모드: 모든 detection을 background로 처리
                # 이렇게 함으로써 추론 모드에서는 refined_obj로 인한 손실을 계산하지 않아도 된다.
                abs_idx_out_ota = conf_idx[nms_out_index]
                abs_idx = abs_idx_out_ota.cpu()
                bg_mask = torch.zeros_like(abs_idx_out_ota).cpu()  # 모두 background
                obj_mask = torch.cat((obj_mask.type_as(bg_mask), bg_mask.unsqueeze(1)))

            # === STEP 4-8: NMS 결과 적용 ===
            detections = detections[nms_out_index]  # NMS 통과한 detection들만 유지

            # === STEP 4-9: 결과 통합 ===
            # 현재 이미지의 detection들을 전체 출력에 추가
            if output[i] is None:
                output[i] = detections  # 첫 번째 detection들
            else:
                output[i] = torch.cat((output[i], detections))  # 기존 OTA detection과 합침

            # 인덱스 정보도 동일하게 통합
            if output_index[i] is None:
                if self.kwargs.get('use_pre_nms',True):
                    output_index[i] = conf_idx[nms_out_index]  # NMS 후 인덱스
                else:
                    output_index[i] = abs_idx  # 절대 인덱스
            else:
                if abs_idx.shape[0] != 0:
                    output_index[i] = torch.cat((output_index[i], abs_idx))  # 기존과 합침

            # 현재 이미지의 객체 마스크 저장
            refined_obj_masks.append(obj_mask)

        # === STEP 5: 전체 배치 결과 통합 ===
        if len(refined_obj_masks) > 0:
            refined_obj_masks = torch.cat(refined_obj_masks, 0)  # 모든 이미지의 마스크 합침
        else:
            refined_obj_masks = torch.zeros(0,1)  # 빈 마스크

        # === STEP 6: 최종 결과 반환 ===
        # output: 각 이미지별 선별된 고품질 proposal들
        # output_index: proposal들의 원본 앵커 인덱스들  
        # refined_obj_masks: refined 손실 계산용 fg/bg 마스크
        # reorder_cls: OTA 매칭을 위한 클래스 재정렬 정보
        return output, output_index, refined_obj_masks, reorder_cls

    def get_idx_predictions(self,prediction,idxs,num_classes):
        """
        특정 인덱스들의 예측 결과만 추출하는 헬퍼 함수
        
        Args:
            prediction: 전체 예측 결과들 [batch, total_anchors, 5+classes]
            idxs: 추출할 인덱스들 [batch] of indices
            num_classes: 클래스 수
            
        Returns:
            output: 선별된 예측들 [batch] of [N_selected, 7+classes]
                   형태: [x1,y1,x2,y2,obj_conf,cls_conf,cls_pred,all_cls_scores]
                   
        처리 과정:
        1. center 좌표 -> corner 좌표 변환
        2. 최고 신뢰도 클래스 찾기
        3. 지정된 인덱스들의 정보만 추출
        4. 표준 detection 형태로 구성
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)
            output[i] = detections[idxs[i], :]
        return output

    def get_fg_idx(self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        """
        전체 배치에 대해 OTA 라벨 할당을 수행하는 메인 함수
        
        이 함수는 훈련시 GT 라벨과 예측을 매칭하여 학습에 필요한 타겟들을 생성합니다.
        각 배치의 이미지별로 OTA 알고리즘을 적용하여 최적의 앵커-GT 매칭을 찾습니다.
        
        Args:
            imgs: 입력 이미지들 [batch, 3, H, W]
            x_shifts, y_shifts: 그리드 좌표들 [batch] of [1, total_anchors]
            expanded_strides: stride 정보 [batch] of [1, total_anchors]
            labels: GT 라벨들 [batch, max_objects, 5] (cls,x,y,w,h)
            outputs: 예측 출력들 [batch, total_anchors, 5+classes]
            origin_preds: L1 손실용 원본 예측들
            dtype: 텐서 데이터 타입
            
        Returns:
            fg_ids: 각 배치별 foreground 앵커 인덱스들
            cls_targets: 분류 타겟들 [batch] of one-hot vectors
            reg_targets: 회귀 타겟들 [batch] of bbox coordinates
            obj_targets: objectness 타겟들 [batch] of binary masks
            fg_masks: foreground 마스크들 [batch] of boolean masks
            num_fg: 총 foreground 앵커 수
            num_gts: 총 GT 객체 수
            l1_targets: L1 손실용 타겟들
            
        처리 과정:
        1. 각 배치별로 GT 수 계산
        2. GT가 있는 경우 OTA 할당 수행
        3. 매칭 결과를 기반으로 학습 타겟 생성
        4. 분류는 IoU 가중 one-hot, 회귀는 GT 좌표 사용
        5. 모든 배치의 결과를 리스트로 수집
        """
        # === STEP 1: 예측 출력 분리 및 전처리 ===
        # 모델 출력에서 각 예측 성분을 분리
        bbox_preds = outputs[:, :, :4]  # [batch, total_anchors, 4] - bbox 회귀 예측 (x,y,w,h)
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, total_anchors, 1] - objectness 예측
        cls_preds = outputs[:, :, 5:]  # [batch, total_anchors, num_classes] - 클래스 분류 예측

        # === STEP 2: Ground Truth 라벨 처리 ===
        # Mixup 데이터 증강이 적용되었는지 확인 (라벨 차원이 5보다 크면 mixup)
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]  # mixup일 경우 앞의 5개 차원만 사용 (cls,x,y,w,h)
        else:
            label_cut = labels  # 일반적인 경우 전체 라벨 사용
        
        # 각 배치별 실제 GT 객체 수 계산 (패딩된 0 라벨 제외)
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # [batch] - 각 이미지의 GT 객체 수

        # === STEP 3: 앵커 관련 정보 통합 ===
        total_num_anchors = outputs.shape[1]  # 모든 FPN 레벨의 총 앵커 수
        # 각 FPN 레벨별로 나누어져 있던 그리드 정보들을 하나로 합침
        x_shifts = torch.cat(x_shifts, 1)  # [1, total_anchors] - 모든 앵커의 X 그리드 좌표
        y_shifts = torch.cat(y_shifts, 1)  # [1, total_anchors] - 모든 앵커의 Y 그리드 좌표
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, total_anchors] - 각 앵커의 stride 값
        
        # L1 손실 사용시 원본 예측값들도 통합
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)  # [batch, total_anchors, 4]

        # === STEP 4: 결과 저장용 리스트 초기화 ===
        fg_ids = []          # 각 배치별 foreground 앵커 인덱스들
        cls_targets = []     # 각 배치별 분류 타겟들 (IoU 가중 one-hot)
        reg_targets = []     # 각 배치별 회귀 타겟들 (GT bbox 좌표)
        l1_targets = []      # 각 배치별 L1 손실용 타겟들
        obj_targets = []     # 각 배치별 objectness 타겟들 (binary mask)
        fg_masks = []        # 각 배치별 foreground 마스크들

        num_fg = 0.0         # 전체 배치의 총 foreground 앵커 수
        num_gts = 0.0        # 전체 배치의 총 GT 객체 수

        # === STEP 5: 배치별 OTA 라벨 할당 수행 ===
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])  # 현재 이미지의 GT 객체 수
            num_gts += num_gt
            
            # GT 객체가 없는 경우: 모든 앵커를 background로 설정
            if num_gt == 0:
                fg_idx = []  # foreground 앵커 없음
                cls_target = outputs.new_zeros((0, self.num_classes))  # 빈 분류 타겟
                reg_target = outputs.new_zeros((0, 4))  # 빈 회귀 타겟
                l1_target = outputs.new_zeros((0, 4))   # 빈 L1 타겟
                obj_target = outputs.new_zeros((total_num_anchors, 1))  # 모든 앵커가 background
                fg_mask = outputs.new_zeros(total_num_anchors).bool()   # 모든 앵커가 background

            # GT 객체가 있는 경우: OTA 알고리즘으로 최적 매칭 수행
            else:
                # 현재 이미지의 GT 정보 추출
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # [num_gt, 4] - GT bbox 좌표
                gt_classes = labels[batch_idx, :num_gt, 0]  # [num_gt] - GT 클래스 라벨
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [total_anchors, 4] - 현재 이미지의 bbox 예측

                # OTA 동적 라벨 할당 수행 (메모리 부족시 CPU 모드로 fallback)
                try:
                    # GPU에서 OTA 할당 시도
                    (
                        gt_matched_classes,      # [num_fg] - 매칭된 GT 클래스들
                        fg_mask,                 # [total_anchors] - foreground 마스크
                        pred_ious_this_matching, # [num_fg] - 매칭된 예측의 IoU 점수
                        matched_gt_inds,         # [num_fg] - 매칭된 GT 인덱스들
                        num_fg_img,              # 현재 이미지의 foreground 앵커 수
                    ) = self.get_assignments(  # OTA 알고리즘 호출
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
                    # 메모리 부족으로 실패시 경고 메시지 출력 후 CPU 모드로 재시도
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()  # GPU 메모리 정리
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # CPU 모드로 OTA 재실행
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
                        "cpu",  # CPU 모드 지정
                    )

                torch.cuda.empty_cache()  # 메모리 정리
                num_fg += num_fg_img  # 전체 foreground 앵커 수에 누적

                # === STEP 6: 학습 타겟 생성 ===
                fg_idx = torch.where(fg_mask)[0]  # foreground 앵커들의 인덱스 추출

                # IoU 가중 분류 타겟 생성: one-hot 벡터에 IoU 점수를 곱해서 soft target 생성
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)  # [num_fg, num_classes]
                
                # Objectness 타겟: foreground 마스크 그대로 사용
                obj_target = fg_mask.unsqueeze(-1)  # [total_anchors, 1]
                
                # 회귀 타겟: 매칭된 GT bbox들 사용
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # [num_fg, 4]

                # L1 손실 사용시: 정규화된 L1 타겟 생성
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),  # 초기값 (사용 안됨)
                        gt_bboxes_per_image[matched_gt_inds],  # 매칭된 GT bbox
                        expanded_strides[0][fg_mask],  # foreground 앵커들의 stride
                        x_shifts=x_shifts[0][fg_mask],  # foreground 앵커들의 X 좌표
                        y_shifts=y_shifts[0][fg_mask],  # foreground 앵커들의 Y 좌표
                    )

            # === STEP 7: 현재 배치 결과를 리스트에 저장 ===
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            fg_ids.append(fg_idx)

        # === STEP 8: 최종 정규화 및 반환 ===
        # 분모가 0이 되는 것을 방지하기 위해 최소값 1로 설정
        num_fg = max(num_fg, 1)

        return fg_ids,cls_targets,reg_targets,obj_targets,fg_masks,num_fg,num_gts,l1_targets

    def get_iou_based_label(self,pred_result,idx,labels,outputs,reg_targets,cls_targets):
        """
        IoU 기반으로 refined 라벨을 생성하는 함수
        
        YOLOV++의 개선된 손실 계산을 위해 시간적 집합 후의 예측과 GT 간의
        IoU를 계산하여 새로운 라벨을 할당합니다. 이는 더 정확한 학습 신호를 제공합니다.
        
        Args:
            pred_result: 1차 예측 결과들 [batch] of [N, 7+classes]
            idx: 선별된 proposal 인덱스들 [batch] of indices
            labels: GT 라벨들 [batch, max_objects, 5]
            outputs: 기본 예측 출력들
            reg_targets: 기존 회귀 타겟들
            cls_targets: 기존 분류 타겟들
            
        Returns:
            refine_cls_targets: 새로운 분류 타겟들 [batch] of [N_selected, classes]
            refine_cls_masks: 분류 감독 마스크들 [batch] of boolean masks
            refine_obj_targets: 새로운 objectness 타겟들 [batch] of [N_selected, 1]
            refine_obj_masks: objectness 감독 마스크들 [batch] of boolean masks
            
        IoU 기반 라벨링 규칙:
        - IoU >= 0.6: Positive (foreground), IoU 값으로 가중치 적용
        - IoU < 0.3: Negative (background)
        - 0.3 <= IoU < 0.6: Ignore (감독하지 않음)
        
        이 방식의 장점:
        1. 예측 품질에 따른 적응적 라벨링
        2. 높은 IoU일수록 강한 학습 신호
        3. 애매한 경우 감독 제외로 안정적 학습
        4. 시간적 집합 후 개선된 예측에 맞는 타겟 제공
        """
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
        refine_cls_targets = []
        refine_cls_masks = []
        refine_obj_targets = []
        refine_obj_masks = []
        for batch_idx in range(len(pred_result)):
            num_gt = int(nlabel[batch_idx])
            reg_target = reg_targets[batch_idx]
            if idx[batch_idx] is None: continue
            if num_gt == 0:
                # GT 객체가 없는 경우 처리 (idx[batch_idx]가 None인 조건 처리 필요)
                refine_cls_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                refine_cls_target[:, -1] = 1  # 감독 없음을 나타내는 플래그로 1 설정
                refine_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 2))
            else:
                refine_cls_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                refine_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 2))

                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_result[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                cls_target = cls_targets[batch_idx]

                refine_obj_target[:, -1] = 1   # 감독 없음을 나타내는 플래그로 1 설정
                refine_cls_target[:, -1] = 1   # 감독 없음을 나타내는 플래그로 1 설정
                refine_obj_target = refine_obj_target.type_as(reg_target)
                refine_cls_target = refine_cls_target.type_as(cls_target)

                fg_cls_coord = torch.where(max_iou.values >= 0.6)[0]
                bg_coord = torch.where(max_iou.values < 0.3)[0]
                fg_cls_max_idx = max_iou.indices[fg_cls_coord]
                cls_target_onehot = (cls_target > 0).type_as(cls_target)

                fg_ious = max_iou.values[fg_cls_coord].unsqueeze(-1)
                fg_ious = fg_ious.type_as(cls_target)
                refine_cls_target[fg_cls_coord, :self.num_classes] = cls_target_onehot[fg_cls_max_idx, :] * fg_ious
                refine_cls_target[fg_cls_coord,-1] = 0

                refine_obj_target[fg_cls_coord,0] = 1
                refine_obj_target[fg_cls_coord,-1] = 0
                refine_obj_target[bg_coord,0] = 0
                refine_obj_target[bg_coord, -1] = 0

                # for ele_idx, ele in enumerate(idx[batch_idx]):
                #     if max_iou.values[ele_idx] >= 0.6:
                #         max_idx = int(max_iou.indices[ele_idx])
                #         refine_cls_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                #         refine_obj_target[ele_idx,0] = 1
                #     else:
                #         if 0.6>max_iou.values[ele_idx]>0.3:#follow faster rcnn, <0.3 set to bg, >0.6 set to fg, in between set to ignore
                #             refine_obj_target[ele_idx,-1] = 1
                #         refine_cls_target[ele_idx, -1] = 1 # set no supervision to 1 as flag
            refine_cls_targets.append(refine_cls_target[:, :-1])
            refine_obj_targets.append(refine_obj_target[:, :-1])
            refine_cls_masks.append(refine_cls_target[:,-1]==0)
            refine_obj_masks.append(refine_obj_target[:,-1]==0)
        return refine_cls_targets,refine_cls_masks,refine_obj_targets,refine_obj_masks


"""
=== YOLOV++ Head 네트워크 전체 요약 ===

이 파일은 YOLOV++의 핵심 혁신인 시간적 특징 집합을 구현하는 Head 네트워크입니다.

## 주요 구성 요소

### 1. 초기화 (__init__)
- 기본 YOLOX 구조: cls_convs, reg_convs, cls_preds, reg_preds, obj_preds
- 비디오 특화 추가: cls_convs2, reg_convs2 (vid_cls=True 시)
- 시간적 집합 모듈: MSA_yolov 또는 LocalAggregation 선택
- 분리된 회귀: agg_iou (decouple_reg=True 시)

### 2. Forward Pass 6단계

#### STEP 1: 기본 특징 추출 및 예측
- 각 FPN 레벨에서 stem -> cls_conv/reg_conv -> pred_heads
- 비디오 특화 레이어(cls_convs2) 추가 특징 추출
- 기본 YOLOX 예측 수행

#### STEP 2: 기본 후처리 및 Proposal 선별
- 다중 스케일 출력 합치기 및 디코딩
- OTA 라벨 할당 (훈련 시)
- postprocess_widx를 통한 고품질 proposal 선별 (핵심!)

#### STEP 3: 시간적 특징 집합 준비
- find_feature_score로 선별된 proposal들의 특징/점수 추출
- 배치 차원 확장 및 데이터 타입 통일

#### STEP 4: 특징 집합 방식 선택 (핵심 혁신!)
- MSA (Multi-Scale Attention): 프레임 간 유사도 기반 적응적 융합
- LocalAggregation: 지역적 시간 정보 활용
- Decoupled Regression: 분류/회귀 별도 집합

#### STEP 5: 손실 계산 (훈련 시)
- 기본 YOLOX 손실: iou_loss, obj_loss, cls_loss, l1_loss
- YOLOV++ 개선 손실: loss_refined_cls, loss_refined_iou, loss_refined_obj
- 총 9개 손실 항목 계산

#### STEP 6: 추론 시 최종 후처리
- postprocess로 NMS 등 후처리 수행
- 최종 탐지 결과 반환

## 핵심 혁신점

### 1. 시간적 특징 집합
- 단일 프레임 → 다중 프레임 정보 활용
- MSA를 통한 프레임 간 유사도 계산
- 고품질 proposal만 선별하여 효율성 확보

### 2. 적응적 Proposal 선별
- postprocess_widx에서 1차 필터링
- 각 프레임당 최대 30개(Afternum) proposal 유지
- confidence 기반 dynamic filtering

### 3. 분리된 처리 브랜치
- 분류: cls_convs2 + MSA + cls_pred
- 회귀: reg_convs + agg_iou + obj_pred (decouple_reg=True 시)
- 각 태스크에 특화된 특징 학습

### 4. 개선된 손실 함수
- 시간적 집합 후 refined 손실들 추가
- IoU 기반 라벨 재할당으로 정확도 향상

## 주요 파라미터

### 네트워크 구조
- Afternum (30): 각 프레임당 최종 proposal 수
- sim_thresh (0.75): 프레임 간 유사도 임계값  
- heads (4): Multi-head attention head 수
- width (1.0): 네트워크 폭 스케일링

### 모드 설정
- agg_type: 'msa' (주요) 또는 'localagg'
- ota_mode: OTA 라벨 할당 사용 여부
- reconf: 재신뢰도 계산 여부
- decouple_reg: 분리된 회귀 브랜치 사용 여부
- vid_cls (True): 비디오 분류 레이어 사용
- vid_reg (False): 비디오 회귀 레이어 사용

### 프레임 처리
- gmode (True): 글로벌 프레임 샘플링
- lframe (0): 로컬 연속 프레임 수 - 실제로는 미사용
- gframe (32): 글로벌 샘플링 프레임 수

## 성능 특징
- ImageNet VID에서 92.9 AP50 달성
- 시간적 정보 활용으로 YOLOX 대비 성능 향상
- 효율적인 proposal 선별로 계산량 절약

이 구조를 통해 YOLOV++는 비디오 객체 탐지에서 SOTA 성능을 달성했습니다.
"""