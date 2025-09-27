# YOLO 비디오 객체 탐지를 위한 후처리 및 트랜스포머 기반 어텐션 모듈
# 이 파일은 시간적 정보를 활용한 객체 추적과 다중 스케일 어텐션(MSA) 메커니즘을 구현

import copy

import torch
import torch.nn as nn
from .weight_init import trunc_normal_  # 가중치 초기화 함수
from .losses import IOUloss  # IoU 손실 함수
from torch.nn import functional as F
from matplotlib import pyplot as plt  # 어텐션 시각화용
from yolox.utils.box_op import (box_cxcywh_to_xyxy, generalized_box_iou,extract_position_matrix,
                                extract_position_embedding,
                                pure_position_embedding)  # 바운딩 박스 연산 유틸리티
from yolox.utils import bboxes_iou  # 바운딩 박스 IoU 계산
from yolox.data.datasets.vid import get_timing_signal_1d  # 시간 임베딩 생성
from yolox.models.post_process import get_linking_mat  # 객체 연결 매트릭스
import time
def visual_attention(data):
    """
    어텐션 맵을 시각화하는 함수
    Args:
        data: 어텐션 가중치 텐서
    """
    data = data.cpu()
    data = data.detach().numpy()

    plt.xlabel('x')
    plt.ylabel('score')
    plt.imshow(data)  # 어텐션 맵을 히트맵으로 표시
    plt.show()


def get_position_embedding(rois1, rois2, feat_dim=64):
    """
    두 ROI 집합 간의 위치 임베딩을 계산하는 함수
    Args:
        rois1: 첫 번째 ROI 집합 [num_rois, 4]
        rois2: 두 번째 ROI 집합 [num_ref_rois, 4]  
        feat_dim: 임베딩 차원 (기본값: 64)
    Returns:
        position_embedding: 위치 임베딩 [1, feat_dim, num_rois, num_ref_rois]
    """
    # [num_rois, num_ref_rois, 4] - ROI 간 상대적 위치 행렬 계산
    position_matrix = extract_position_matrix(rois1, rois2)
    # [num_rois, num_ref_rois, 64] - 위치 행렬을 임베딩으로 변환
    position_embedding = extract_position_embedding(position_matrix, feat_dim=feat_dim)
    # [64, num_rois, num_ref_rois] - 차원 순서 변경
    position_embedding = position_embedding.permute(2, 0, 1)
    # [1, 64, num_rois, num_ref_rois] - 배치 차원 추가
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding

class SelfAttentionLocal(nn.Module):
    """
    위치 정보와 시간 정보를 활용한 로컬 자기 어텐션 모듈
    비디오 객체 탐지에서 시공간적 관계를 모델링
    """
    def __init__(self, dim, num_heads=8, bias=False, attn_drop=0.,**kwargs):
        super().__init__()
        self.num_heads = num_heads  # 멀티헤드 어텐션의 헤드 수
        head_dim = dim // num_heads  # 각 헤드의 차원
        # NOTE: 스케일 팩터 (어텐션 스코어 정규화용)
        self.scale = head_dim ** -0.5
        
        # 다양한 설정 옵션들
        self.use_time_emd = kwargs.get('use_time_emd', False)  # 시간 임베딩 사용 여부
        self.use_loc_emb = kwargs.get('use_loc_emd', True)     # 위치 임베딩 사용 여부
        self.loc_fuse_type = kwargs.get('loc_fuse_type', 'add')  # 위치 정보 융합 방식
        self.use_qkv = kwargs.get('use_qkv', True)             # QKV 사용 여부
        self.locf_dim = kwargs.get('locf_dim', 64)             # 위치 특성 차원
        self.loc_emd_dim = kwargs.get('loc_emd_dim', 64)       # 위치 임베딩 차원
        self.pure_pos_emb = kwargs.get('pure_pos_emb', False)  # 순수 위치 임베딩 사용
        self.loc_conf = kwargs.get('loc_conf', False)          # 위치 신뢰도 사용
        self.iou_base = kwargs.get('iou_base', False)          # IoU 기반 어텐션 사용
        self.iou_thr = kwargs.get('iou_thr', 0.5)              # IoU 임계값
        self.reconf = kwargs.get('reconf', False)              # 분류/회귀 분리 설정
        self.iou_window = kwargs.get('iou_window', 0)          # IoU 윈도우 크기
        # IoU 기반 어텐션 사용 시 다른 임베딩들 비활성화
        if self.iou_base:
            self.use_time_emd = False
            self.use_loc_emb = False
            self.pure_pos_emb = False

        # 분류/회귀 분리 설정에 따른 레이어 구성
        if self.reconf:
            # 분류와 회귀를 분리해서 처리
            self.qk = nn.Linear(dim * 2, dim * 4, bias=bias)  # Query, Key 생성
            self.v_cls = nn.Linear(dim, dim, bias=bias)       # 분류용 Value
            self.v_reg = nn.Linear(dim, dim, bias=bias)       # 회귀용 Value
        else:
            # 통합된 QKV 처리
            self.qkv = nn.Linear(dim, dim * 3, bias=bias)

        # 위치 임베딩 관련 레이어들
        if self.use_loc_emb:
            if self.pure_pos_emb:
                # 순수 위치 임베딩: 4차원 좌표를 특성으로 변환
                self.loc2feature = nn.Linear(4, dim, bias=False)
                self.loc_fuse_type = 'identity'
            else:
                # 위치 특성을 어텐션 헤드 수만큼의 채널로 변환
                self.loc2feature = nn.Conv2d(self.locf_dim, self.num_heads,kernel_size=1,stride=1,padding=0)
                # 위치 특성 변환 레이어 초기화
                trunc_normal_(self.loc2feature.weight, std=0.01)
                nn.init.constant_(self.loc2feature.bias, 0)
                self.locAct = nn.ReLU(inplace=True)  # 활성화 함수
        
        self.attn_drop = nn.Dropout(attn_drop)  # 어텐션 드롭아웃

    def forward(self,x, x_reg, locs, **kwargs):
        """
        Self-Attention 순전파
        Args:
            x: 분류 특성 [B, N, C]
            x_reg: 회귀 특성 [B, N, C] 
            locs: 바운딩 박스 위치 [LF, P, 4]
            **kwargs: 추가 파라미터들
        """
        B, N, C = x.shape  # 배치 크기, 객체 수, 특성 차원
        L,G,P = kwargs.get('lframe'),kwargs.get('gframe'),kwargs.get('afternum')  # 로컬/글로벌 프레임, 후보 수

        assert B == 1 , 'only support video batch size 1 currently'  # 현재 배치 크기 1만 지원
        if locs!=None:
            LF, P= locs.shape[0], locs.shape[1]  # 프레임 수, 객체 수 추출
            locs = locs.view(-1, 4)  # [LF*P, 4]로 reshape

        if self.use_loc_emb and not self.pure_pos_emb:
            loc_emd = get_position_embedding(locs,locs,feat_dim=self.loc_emd_dim).type_as(x) #1, 64, N, N
            if self.use_time_emd:
                time_emd = get_timing_signal_1d(torch.arange(0,LF), self.locf_dim).type_as(x) #LF, 64
                time_emd = time_emd.unsqueeze(1).repeat(P, N, 1).permute(2, 0, 1).unsqueeze(0) #1, 64, N, N
                loc_time_emb = loc_emd + time_emd
            else:
                loc_time_emb = loc_emd
            attn_lt = self.locAct(self.loc2feature(loc_time_emb))#
            fg_score = kwargs.get('fg_score', None)
            if self.loc_conf and fg_score is not None:
                fg_score = fg_score>0.001
                fg_score = fg_score.view(1,-1).unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, N, 1)
                fg_score = fg_score.type_as(x)
                attn_lt = attn_lt * fg_score
        elif self.pure_pos_emb:
            pure_loc_features = pure_position_embedding(locs,kwargs.get('width'),kwargs.get('height')).type_as(x)
            pure_loc_features = self.loc2feature(pure_loc_features)#B*N,C
            pure_loc_features = pure_loc_features.view(B, N, C)
            if self.use_time_emd:
                time_emd = get_timing_signal_1d(torch.arange(0, LF), C).type_as(x)  # LF, C
                time_emd = time_emd.unsqueeze(0).repeat(B, P, 1)  # B, N, C
                pure_loc_features = pure_loc_features + time_emd
            x = x + pure_loc_features.reshape(B,N,-1)
        elif self.iou_base:
            if self.iou_window != 0:
                iou_masks = torch.zeros((N, N))
                for i in range(L):
                    lower = max(i - self.iou_window, 0)
                    upper = min(i + self.iou_window, L)
                    iou_masks[lower * P:upper * P, i * P:(i + 1) * P] = 1
                iou_masks = iou_masks.type_as(x)
            else:
                iou_masks = 1
                # set the
            iou_mat = bboxes_iou(locs, locs)  # N,N
            iou_mat = (iou_mat > 0.0)*iou_masks

        if self.reconf:
            qk = self.qk(torch.cat([x,x_reg],dim=-1))
            qk = qk.reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qk[0], qk[1]
            v_cls = self.v_cls(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v_loc = self.v_reg(x_reg).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v_cls = qkv[0], qkv[1], qkv[2] # B, num_heads, N, C
            # qk = self.qk(torch.cat([x, x_reg], dim=-1))
            # qk = qk.reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # q, k = qk[0], qk[1]
            # v_cls = self.v_cls(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)


        attn = (q @ k.transpose(-2, -1)) * self.scale #B, num_heads, M, N
        cls_score = kwargs.get('cls_score', None)
        if self.loc_conf and cls_score is not None:
            cls_score = cls_score.view(1,-1).unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, N, 1)
            cls_score = cls_score.type_as(x)
            attn = attn * cls_score

        if self.loc_fuse_type == 'add' and not self.iou_base:
            attn = attn + (attn_lt+1e-6).log()
        elif self.loc_fuse_type == 'dot' and not self.iou_base:
            attn = attn * (attn_lt+1e-6).log()
        elif self.loc_fuse_type == 'identity' or self.iou_base:
            attn = attn
        else:
            raise NotImplementedError
        attn = attn.softmax(dim=-1)
        if self.iou_base:
            attn = attn*iou_mat.type_as(x)
            attn = attn / (torch.sum(attn, dim=-1, keepdim=True))
        attn = self.attn_drop(attn)

        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        if self.reconf:
            x_reg = (attn @ v_loc).transpose(1, 2).reshape(B, N, C)
            return x, x_reg
        return x

class FFN(nn.Module):
    """
    Feed-Forward Network (FFN) 모듈
    트랜스포머의 MLP 부분을 구현
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),      # 입력 차원을 은닉 차원으로 확장
            nn.GELU(),                       # GELU 활성화 함수
            nn.Dropout(dropout),             # 드롭아웃
            nn.Linear(hidden_dim, dim),      # 은닉 차원을 다시 입력 차원으로 축소
            nn.Dropout(dropout)              # 드롭아웃
        )
    
    def forward(self, x):
        """
        FFN 순전파
        Args:
            x: 입력 특성 [B, N, dim]
        Returns:
            출력 특성 [B, N, dim]
        """
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    트랜스포머 블록 모듈
    Self-Attention + FFN으로 구성되며, 분류/회귀 분리 옵션 지원
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False, dropout=0., attn_drop=0., drop_path=0.,**kwargs):
        super().__init__()
        # 정규화 레이어들
        self.norm1 = nn.LayerNorm(dim)           # 어텐션 전 정규화 (분류용)
        self.norm3 = nn.LayerNorm(dim)           # 어텐션 전 정규화 (회귀용)
        
        # Self-Attention 모듈
        self.attn = SelfAttentionLocal(dim, num_heads=num_heads, bias=qkv_bias, attn_drop=attn_drop, **kwargs)
        self.drop_path = nn.Identity()           # DropPath (현재는 Identity로 설정)
        
        # 설정 옵션들
        self.use_ffn = kwargs.get('use_ffn', True)     # FFN 사용 여부
        self.reconf = kwargs.get('reconf', False)      # 분류/회귀 분리 여부
        
        # FFN 관련 레이어들
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(dim)       # FFN 전 정규화 (분류용)
            self.mlp = FFN(dim, int(dim * mlp_ratio), dropout=dropout)  # 분류용 FFN
            if self.reconf:
                self.norm4 = nn.LayerNorm(dim)   # FFN 전 정규화 (회귀용)
                self.mlp_conf = FFN(dim, int(dim * mlp_ratio), dropout=dropout)  # 회귀용 FFN
    def forward(self, x, x_reg, locs,**kwargs):
        if self.reconf:
            x_cls,x_reg_ = self.attn(self.norm1(x),self.norm3(x_reg), locs, **kwargs)
            x_reg = x_reg + self.drop_path(x_reg_)
            x_cls = x_cls + self.drop_path(x)
            if self.use_ffn:
                x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
                x_reg = x_reg + self.drop_path(self.mlp_conf(self.norm4(x_reg)))
            return x_cls, x_reg
        else:
            x = x + self.drop_path(self.attn(self.norm1(x),self.norm3(x_reg), locs, **kwargs))
            if self.use_ffn:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, x_reg
class Attention_msa(nn.Module):
    """
    Multi-Scale Attention (MSA) 모듈
    분류와 회귀 특성을 분리해서 처리하는 어텐션 메커니즘
    비디오 객체 탐지에서 시간적 일관성을 유지하기 위해 사용
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        """
        Args:
            dim: 입력 특성 차원
            num_heads: 멀티헤드 어텐션의 헤드 수
            qkv_bias: QKV 선형 변환에 bias 사용 여부
            attn_drop: 어텐션 드롭아웃 비율
            scale: 어텐션 스케일 팩터 (기본값: 25)
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE: 스케일 팩터 설정 (기존 가중치와의 호환성을 위해 수동 설정 가능)
        self.scale = scale  # qk_scale or head_dim ** -0.5

        # 분류와 회귀를 위한 별도의 QKV 변환
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 분류용 QKV
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 회귀용 QKV

        self.attn_drop = nn.Dropout(attn_drop)  # 어텐션 드롭아웃

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,
                return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False,**kwargs):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        attn_reg_raw = v_reg_normed @ v_reg_normed.transpose(-2, -1)

        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1



        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        #remove ave and conf guide in the reg branch, modified in 2023.12.5
        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale# * fg_score * fg_score_mask

        if kwargs.get('local_mask', False):
            lframe,gframe,P = kwargs.get('lframe'),kwargs.get('gframe'),kwargs.get('afternum')
            local_mask_branch = kwargs.get('local_mask_branch')
            if 'cls' in local_mask_branch:
                attn_cls[:, :, 0:lframe * P, 0:lframe * P] = -1e4
            if 'reg' in local_mask_branch:
                attn_reg[:, :, 0:lframe * P, 0:lframe * P] = -1e4

        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        #
        x_reg = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)
        x_ori_reg = v_reg.permute(0, 2, 1, 3).reshape(B, N, C)
        x_reg = torch.cat([x_reg, x_ori_reg], dim=-1)

        if ave:
            conf_sim_thresh = kwargs.get('conf_sim_thresh', 0.99)
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            #remove ave and conf guide in the reg branch, modified in 2023.12.5
            obj_mask = torch.where(attn_reg_raw > conf_sim_thresh, ones_matrix, zero_matrix)
            if use_mask:
                sim_mask = sim_mask*cls_score_mask[0,0,:,:]*fg_score_mask[0,0,:,:]
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            obj_mask = obj_mask * sim_round2 / (torch.sum(obj_mask * sim_round2, dim=-1, keepdim=True))
            return x_cls, x_reg, sim_round2, obj_mask
        else:
            return x_cls, x_reg, None, None

class Attention_msa_visual(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = 30#scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,img = None, pred = None):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls,dim=-1,keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score,[1,1,1,-1]).repeat(1,self.num_heads,N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1,self.num_heads,N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score #* cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_cls_raw*25).softmax(dim=-1)#attn_cls#(attn_reg + attn_cls) / 2 #attn_reg#(attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0,2,1,3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)

        ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
        zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

        attn_cls_raw = torch.sum(attn_cls_raw,dim=1,keepdim=False)[0] / self.num_heads
        sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
        sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

        sim_round2 = torch.softmax(sim_attn, dim=-1)
        sim_round2 = sim_mask*sim_round2/(torch.sum(sim_mask*sim_round2,dim=-1,keepdim=True))
        from yolox.models.post_process import visual_sim
        attn_total = torch.sum(attn,dim=1,keepdim=False)[0] / self.num_heads
        visual_sim(attn_total,img,30,pred,attn_cls_raw)
        return x_cls,None,sim_round2

class Attention_msa_online(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False,ave = True):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls,dim=-1,keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score,[1,1,1,-1]).repeat(1,self.num_heads,N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1,self.num_heads,N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0,2,1,3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw,dim=1,keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask*sim_round2/(torch.sum(sim_mask*sim_round2,dim=-1,keepdim=True))
            return x_cls,None,sim_round2
        else:
            return x_cls


class LocalAggregation(nn.Module):
    def __init__(self,dim,heads,bias=False,attn_drop=0., blocks=1,**kwargs):
        super().__init__()
        self.blocks = blocks
        self.transBlocks = nn.ModuleList()
        for i in range(blocks):
            self.transBlocks.append(TransformerBlock(dim,heads,qkv_bias=bias,attn_drop=attn_drop,**kwargs))
    def forward(self,x,x_reg,locs=None,**kwargs):
        for i in range(self.blocks):
            x,x_reg = self.transBlocks[i](x,x_reg,locs,**kwargs)
        return x,x_reg


class MSA_yolov(nn.Module):
    """
    YOLO를 위한 Multi-Scale Attention 모듈
    어텐션 결과를 선형 변환을 통해 최종 출력으로 변환
    """
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25, reconf=False):
        super().__init__()
        self.reconf = reconf  # 분류/회귀 분리 설정
        
        # MSA 어텐션 모듈
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        
        # 분류용 선형 변환 레이어들
        self.linear1 = nn.Linear(2 * dim, 2 * dim)    # 첫 번째 변환
        self.linear2 = nn.Linear(4 * dim, out_dim)    # 최종 출력 변환
        
        # 회귀용 별도 레이어 (reconf=True일 때)
        if reconf:
            self.linear1_obj = nn.Linear(2 * dim, 2 * dim)  # 회귀용 첫 번째 변환
            self.linear2_obj = nn.Linear(4 * dim, out_dim)   # 회귀용 최종 출력 변환
    def find_similar_round2(self, features, ave_mask,feature_obj,mask_obj,fg_score=None):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            ave_mask = ave_mask.to(features.dtype)
        soft_sim_feature = (ave_mask @ support_feature)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        if self.reconf:
            mask_obj = mask_obj.to(features.dtype)
            key_feature_obj = feature_obj[0]
            support_feature_obj = feature_obj[0]
            soft_sim_feature_obj = (mask_obj @ support_feature_obj)
            obj_feature = torch.cat([soft_sim_feature_obj, key_feature_obj], dim=-1)
        else:
            obj_feature = None
        return cls_feature,obj_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False,**kwargs):
        trans_cls, trans_obj, ave_mask,obj_mask  = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
                                                    use_mask=use_mask,**kwargs)

        trans_cls = self.linear1(trans_cls) #
        if self.reconf:
            trans_obj = self.linear1_obj(trans_obj)
        trans_cls,trans_obj = self.find_similar_round2(trans_cls, ave_mask, trans_obj, obj_mask)
        trans_cls = self.linear2(trans_cls)
        if self.reconf:
            trans_obj = self.linear2_obj(trans_obj)
        return trans_cls,trans_obj


class MSA_yolov_visual(nn.Module):

    def __init__(self, dim,out_dim, num_heads=4, qkv_bias=False, attn_drop=0.,scale=25):
        super().__init__()
        self.msa = Attention_msa_visual(dim,num_heads,qkv_bias,attn_drop,scale=scale)
        self.linear1 = nn.Linear(2 * dim,2 * dim)
        self.linear2 =  nn.Linear(4 * dim,out_dim)

    def ave_pooling_over_ref(self,features,sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results@support_feature)#.transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature,key_feature],dim=-1)
        return cls_feature

    def forward(self,x_cls, x_reg, cls_score = None, fg_score = None, img = None, pred = None):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls,x_reg,cls_score,fg_score,img,pred)
        msa = self.linear1(trans_cls)
        ave = self.ave_pooling_over_ref(msa,sim_round2)
        out = self.linear2(ave)
        return out

class MSA_yolov_online(nn.Module):
    """
    온라인 비디오 처리를 위한 MSA 모듈
    실시간 객체 추적에서 사용되며, 로컬 집계 기능 포함
    """
    def __init__(self, dim,out_dim, num_heads=4, qkv_bias=False, attn_drop=0.,scale=25):
        super().__init__()
        # 온라인 MSA 어텐션 모듈
        self.msa = Attention_msa_online(dim,num_heads,qkv_bias,attn_drop,scale=scale)
        # 선형 변환 레이어들
        self.linear1 = nn.Linear(2 * dim,2 * dim)    # 첫 번째 변환
        self.linear2 =  nn.Linear(4 * dim,out_dim)   # 최종 출력 변환

    def ave_pooling_over_ref(self,features,sort_results):
        """
        참조 특성들에 대한 가중 평균 풀링
        Args:
            features: 입력 특성 [B, N, dim]
            sort_results: 유사도 가중치 [N, N]
        Returns:
            cls_feature: 키 특성과 지원 특성을 결합한 특성 [B, N, 2*dim]
        """
        key_feature = features[0]      # 현재 프레임 특성
        support_feature = features[0]  # 참조 프레임 특성
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        # 유사도 가중치를 사용한 소프트 집계
        soft_sim_feature = (sort_results@support_feature)
        # 키 특성과 집계된 특성을 결합
        cls_feature = torch.cat([soft_sim_feature,key_feature],dim=-1)

        return cls_feature

    def compute_geo_sim(self,key_preds,ref_preds):
        """
        기하학적 유사도(IoU) 계산
        Args:
            key_preds: 현재 예측 결과 [M, ...]
            ref_preds: 참조 예측 결과 [N, ...]
        Returns:
            iou: IoU 유사도 행렬 [M, N]
        """
        key_boxes = key_preds[:,:4]  # 현재 바운딩 박스 [M, 4]
        ref_boxes = ref_preds[:,:4]  # 참조 바운딩 박스 [N, 4]
        # Generalized IoU 계산
        cost_giou, iou = generalized_box_iou(key_boxes.to(torch.float32), ref_boxes.to(torch.float32))

        return iou.to(torch.float16)

    def local_agg(self,features,local_results,boxes,cls_score,fg_score):
        """
        로컬 특성들과의 집계 함수
        과거 프레임의 지역 결과들을 현재 특성과 융합
        Args:
            features: 현재 특성 [M, dim]
            local_results: 과거 지역 결과들 {'msa', 'boxes', 'cls_scores', 'reg_scores'}
            boxes: 현재 바운딩 박스 [M, 4]
            cls_score: 현재 분류 스코어 [M]
            fg_score: 현재 전경 스코어 [M]
        Returns:
            융합된 특성 [M, dim]
        """
        # 과거 지역 특성들 정규화
        local_features = local_results['msa']
        local_features_n = local_features / torch.norm(local_features, dim=-1, keepdim=True)
        features_n = features /torch.norm(features, dim=-1, keepdim=True)
        # 코사인 유사도 계산
        cos_sim = features_n@local_features_n.transpose(0,1)

        # 기하학적 유사도(IoU) 계산
        geo_sim = self.compute_geo_sim(boxes,local_results['boxes'])
        N = local_results['cls_scores'].shape[0]  # 과거 객체 수
        M = cls_score.shape[0]                    # 현재 객체 수
        
        # 신뢰도 점수 비교를 위한 행렬 생성
        pre_scores = cls_score*fg_score
        pre_scores = torch.reshape(pre_scores, [-1, 1]).repeat(1, N)  # [M, N]
        other_scores = local_results['cls_scores']*local_results['reg_scores']
        other_scores = torch.reshape(other_scores, [1, -1]).repeat(M, 1)  # [M, N]
        
        # 신뢰도 임계값 마스크 생성 (과거 객체가 현재 객체보다 크게 낮지 않은 경우만)
        ones_matrix = torch.ones([M,N]).to('cuda')
        zero_matrix = torch.zeros([M,N]).to('cuda')
        thresh_map = torch.where(other_scores-pre_scores>-0.3,ones_matrix,zero_matrix)
        
        # 최종 유사도: 코사인 유사도 * 임계값 마스크 * 기하학적 유사도
        local_sim = torch.softmax(25*cos_sim*thresh_map,dim=-1)*geo_sim
        local_sim = local_sim / torch.sum(local_sim, dim=-1, keepdim=True)  # 정규화
        local_sim = local_sim.to(features.dtype)
        # 유사도 가중 평균을 통한 특성 융합
        sim_features = local_sim @ local_features

        return (sim_features+features)/2  # 현재 특성과 융합된 특성의 평균

    def forward(self,x_cls, x_reg, cls_score = None, fg_score = None,other_result = {},boxes=None, simN=30):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls,x_reg,cls_score,fg_score)
        msa = self.linear1(trans_cls)
        # if other_result != []:
        #     other_msa = other_result['msa'].unsqueeze(0)
        #     msa = torch.cat([msa,other_msa],dim=1)
        ave = self.ave_pooling_over_ref(msa,sim_round2)
        out = self.linear2(ave)
        if other_result != [] and other_result['local_results'] != []:
            lout = self.local_agg(out[:simN],other_result['local_results'],boxes[:simN],cls_score[:simN],fg_score[:simN])
            return lout,out
        return out,out













