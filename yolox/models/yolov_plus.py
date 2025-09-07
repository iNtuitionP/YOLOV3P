#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

class YOLOV(nn.Module):
    """
    YOLOV++ 메인 모델 클래스
    
    YOLOV++는 비디오 객체 탐지를 위한 모델로, 기존 YOLOX에 시간적 특징 집합(temporal feature aggregation) 기능을 추가했습니다.
    
    구조:
    - backbone: 이미지 특징 추출 (예: CSPDarknet, Swin Transformer, FocalNet)
    - head: 비디오 특징 집합 및 객체 탐지 헤드 (YOLOV++의 핵심 혁신)
    
    주요 특징:
    1. 다중 프레임 입력 처리 (lframe + gframe)
    2. Multi-Scale Attention (MSA)을 통한 시간적 특징 집합
    3. 개선된 손실 함수 (refined classification, IoU, objectness)
    """

    def __init__(self, backbone=None, head=None):
        """
        YOLOV++ 모델 초기화
        
        Args:
            backbone: 특징 추출 백본 네트워크 (YOLOPAFPN, Swin, FocalNet 등)
            head: 비디오 객체 탐지 헤드 (YOLOVHead - MSA 포함)
        """
        super().__init__()
        self.backbone = backbone  # 특징 추출을 담당하는 백본 네트워크
        self.head = head         # 비디오 특징 집합 및 최종 예측을 담당하는 헤드

    def forward(self, x, targets=None, nms_thresh=0.5, lframe=0, gframe=32):
        """
        YOLOV++ Forward Pass
        
        Args:
            x (Tensor): 입력 비디오 프레임 배치 [B, C, H, W]
                       B = lframe + gframe (로컬 프레임 + 글로벌 프레임)
            targets: 훈련 시 Ground Truth 라벨 (추론 시 None)
            nms_thresh (float): Non-Maximum Suppression 임계값 (추론 시 사용)
            lframe (int): 로컬 프레임 수 - 연속된 프레임들로 시간적 연속성 활용
            gframe (int): 글로벌 프레임 수 - 전체 비디오에서 샘플링된 프레임들
            
        Returns:
            dict: 훈련 시 - 손실 값들의 딕셔너리
                  추론 시 - 탐지 결과 (bboxes, scores, classes)
        """
        
        # Step 1: 백본을 통한 다중 스케일 특징 추출
        # FPN 출력: [dark3, dark4, dark5] - 3개의 서로 다른 해상도 특징맵
        # dark3: 높은 해상도 (세밀한 객체), dark4: 중간 해상도, dark5: 낮은 해상도 (큰 객체)
        fpn_outs = self.backbone(x)
        
        if self.training:
            # === 훈련 모드: 손실 계산 ===
            assert targets is not None, "훈련 시에는 Ground Truth targets이 필요합니다"
            
            # YOLOV++ 헤드에서 다중 손실 계산
            # 기존 YOLOX 손실 + YOLOV++ 개선된 손실들
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, \
            loss_refined_cls, \
            loss_refined_iou, \
            loss_refined_obj = self.head(
                fpn_outs, targets, x, lframe=lframe, gframe=gframe
            )
            
            # 훈련 손실들을 딕셔너리로 반환
            outputs = {
                # === 기존 YOLOX 손실들 ===
                "total_loss": loss,              # 전체 손실 (모든 손실의 가중합)
                "iou_loss": iou_loss,           # IoU 회귀 손실 (bounding box 정확도)
                "l1_loss": l1_loss,             # L1 회귀 손실 (bbox 좌표 정밀도)
                "conf_loss": conf_loss,         # Objectness 신뢰도 손실
                "cls_loss": cls_loss,           # 분류 손실
                "num_fg": num_fg,               # Foreground 앵커 수 (디버깅용)
                
                # === YOLOV++ 추가 개선된 손실들 ===
                "loss_refined_cls": loss_refined_cls,    # 시간적 특징 집합 후 개선된 분류 손실
                "loss_refined_iou": loss_refined_iou,    # 시간적 특징 집합 후 개선된 IoU 손실  
                "loss_refined_obj": loss_refined_obj     # 시간적 특징 집합 후 개선된 Objectness 손실
            }
        else:
            # === 추론 모드: 객체 탐지 결과 생성 ===
            # NMS를 포함한 후처리까지 수행하여 최종 탐지 결과 반환
            outputs = self.head(fpn_outs, targets, x, 
                               nms_thresh=nms_thresh, 
                               lframe=lframe, 
                               gframe=gframe)

        return outputs


"""
=== YOLOV++ 모델 구조 상세 분석 ===

1. 입력 데이터 구조:
   - 비디오 프레임들을 배치로 처리: [lframe + gframe, C, H, W]
   - lframe: 연속된 로컬 프레임들 (시간적 연속성 활용) 
        -> lframe 이 있지만 lframe = 0 인 걸로 봐서는 구현하는 과정에서 lframe이 없는 것이 더 좋은 것 같음(2025-09-07 기준)
   - gframe: 글로벌 샘플링된 프레임들 (장기 의존성 활용)

2. 백본 네트워크 (Backbone):
   - CSPDarknet (기본), Swin Transformer, FocalNet 등 지원
   - FPN (Feature Pyramid Network) 구조로 다중 스케일 특징 추출
   - 출력: 3개 레벨의 특징맵 [dark3, dark4, dark5]

3. 헤드 네트워크 (Head) - YOLOV++의 핵심:
   - Multi-Scale Attention (MSA)을 통한 시간적 특징 집합
   - 프레임 간 유사도 계산 및 가중 평균
   - 기존 YOLOX 손실 + 개선된 refined 손실들

4. 주요 차이점 (vs YOLOX):
   - 단일 이미지 → 다중 비디오 프레임 처리
   - 시간적 정보 활용을 통한 성능 향상
   - 비디오 특화 후처리 (trajectory linking 등)

5. 훈련/추론 차이:
   - 훈련: 9개의 서로 다른 손실 값 계산
        -> 최종 loss function은 어떻게 구현되어있는지 확인할 필요가 있음. 9개의 loss를 모두 사용하지는 않을테니까(2025-09-07 기준)
   - 추론: NMS 포함 최종 탐지 결과 반환
"""
