import os
import argparse
import yaml
import cv2
from tqdm import tqdm
from utils import *
import time
import warnings



# ==== 新增：MFRGN Baseline 封装（请根据项目路径调整） ====
import torch
import torch.nn as nn

try:
    # 项目内路径：Retrieval_Models/MFRGN/mfrgn_model.py 中定义的 MFRGNModel 类
    from Retrieval_Models.MFRGN.mfrgn_model import MFRGNModel
except Exception as _e:
    MFRGNModel = None  # 若未成功导入 MFRGNModel，不影响主流程（由 retrieval_init 决定是否使用）

# 微调后的 MFRGN 权重路径（将此处更新为实际的 checkpoint 路径）
# MFRGN_CKPT_PATH =  "checkpoints/university/convnext_base.fb_in22k_ft_in1k/u652-D2S_U1652-D2S_10-30-11-29-46/weights_e1_92.2335.pth"
# MFRGN_CKPT_PATH = "checkpoints/university/convnext_base.fb_in22k_ft_in1k/u652-D2S_U1652-D2S_10-30-11-29-46/weights_end.pth"Recall@1 = 92.2335 11.2 epoch=1

# MFRGN_CKPT_PATH = "checkpoints/university/convnext_base.fb_in22k_ft_in1k/u652-D2S_U1652-D2S_11-11-10-08-15/weights_end.pth"

# MFRGN_CKPT_PATH = "checkpoints/university/convnext_base.fb_in22k_ft_in1k/u652-D2S_U1652-D2S_11-12-21-53-47/weights_end.pth"epoch=20 这里没用epoch=6的best model

MFRGN_CKPT_PATH = "checkpoints/university/convnext_base.fb_in22k_ft_in1k/u652-D2S_U1652-D2S_11-12-21-53-47/weights_e6_92.7302.pth"
 

class Baseline(nn.Module):
    def __init__(self, pretrained_backbone_path: str = None):
        """
        Baseline 封装，用于 MFRGN 模型。可选传入预训练 ConvNeXt 骨干网络权重路径。
        """
        super().__init__()
        if MFRGNModel is None:
            raise ImportError("无法从 Retrieval_Models.MFRGN.mfrgn_model 导入 MFRGNModel 类。")
        # 初始化 MFRGN 模型（内部会处理是否加载 ConvNeXt 预训练权重）
        self.backbone = MFRGNModel(model_name='convnext_base', pretrained_backbone_path=pretrained_backbone_path)
        # Load fine-tuned MFRGN weights, if available
        if MFRGN_CKPT_PATH and os.path.exists(MFRGN_CKPT_PATH):
            state_dict = torch.load(MFRGN_CKPT_PATH, map_location='cpu')
            # Load the weights into the backbone (allow missing keys for logit_scale/norm1)
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded fine-tuned MFRGN weights from {MFRGN_CKPT_PATH}")
        else:
            print("Warning: Fine-tuned MFRGN checkpoint not found, using default weights.")
        # Preserve interface consistency with other models
        self.logit_scale = nn.Parameter(torch.ones([]) * 1.0)
        self.norm1 = nn.LayerNorm(128)




    def forward(self, x):
        x = self.backbone(x)                 # x.shape = [B, 384]
        global_feat = x[:, :128]            # 前128维全局特征
        local_feat = x[:, 128:]            # 后256维局部特征
        global_feat = self.norm1(global_feat)      # 归一化
        global_feat = global_feat * self.logit_scale  # 缩放
        return torch.cat([global_feat, local_feat], dim=1)



# ==== 新增封装结束 ====

warnings.filterwarnings("ignore")


def get_parse():
    parser = argparse.ArgumentParser(description='UAV-Visual-Localization')
    parser.add_argument('--yaml', default='config.yaml', type=str, help='配置 YAML 文件路径')
    parser.add_argument('--save_dir', default='./Result/Experiment1/', type=str, help='结果保存目录')
    parser.add_argument('--device', default='cuda', type=str, help='推理设备 (cuda 或 cpu)')
    parser.add_argument('--pose_priori', default='yp', type=str,
                        help="姿态先验信息: 'yp' (偏航+俯仰), 'p' (仅俯仰), 'unknown' (无先验)")
    parser.add_argument('--strategy', default='Topn_opt', type=str,
                        help="定位策略: 'Inliers'; 'Top1'; 'Topn_opt'; 'wo_retrieval'")
    parser.add_argument('--PnP_method', default='P3P', type=str, help="PnP求解方法（详见 option.yaml 可选项）")
    parser.add_argument('--Ref_type', default='HIGH', type=str,
                        help="使用的参考地图类型: 'HIGH' 表示航拍正射图, 'LOW' 表示卫星图")
    parser.add_argument('--resize_ratio', default=0.2, type=float, help='参考地图缩放比例以加速处理')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    opt = get_parse()
    # 使用 safe_load 读取全局配置（UTF-8防止编码问题）
    with open(opt.yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 创建结果目录（若不存在）
    os.makedirs(opt.save_dir, exist_ok=True)
    print(f"Result saved in : {opt.save_dir}")
    All_Region = config['REGIONS']
    All_Retrieval = config['RETRIEVAL_METHODS']
    All_Matching = config['MATCHING_METHODS']
    # 遍历配置中的每个区域
    for region in All_Region:
        yaml_file = f'./Regions_params/{region}.yaml'
        # 读取区域特定配置
        with open(yaml_file, 'r', encoding='utf-8') as f:
            region_config = yaml.safe_load(f)
        # 将全局配置合并进区域配置
        region_config.update(config)
        # 如果数据在 UAV_AVL_demo 子目录，则调整路径
        if not os.path.exists('./Data') and os.path.exists('./UAV_AVL_demo/Data'):
            if 'UAV_PATH' in region_config and not os.path.exists(region_config['UAV_PATH']):
                alt_path = os.path.join('UAV_AVL_demo', region_config['UAV_PATH'].lstrip("./\\"))
                if os.path.exists(alt_path):
                    region_config['UAV_PATH'] = alt_path
            if 'HIGH_RES_MAP' in region_config and not os.path.exists(region_config['HIGH_RES_MAP']):
                alt_path = os.path.join('UAV_AVL_demo', region_config['HIGH_RES_MAP'].lstrip("./\\"))
                if os.path.exists(alt_path):
                    region_config['HIGH_RES_MAP'] = alt_path
            if 'LOW_RES_MAP' in region_config and not os.path.exists(region_config['LOW_RES_MAP']):
                alt_path = os.path.join('UAV_AVL_demo', region_config['LOW_RES_MAP'].lstrip("./\\"))
                if os.path.exists(alt_path):
                    region_config['LOW_RES_MAP'] = alt_path
            if 'DSM_MAP' in region_config and not os.path.exists(region_config['DSM_MAP']):
                alt_path = os.path.join('UAV_AVL_demo', region_config['DSM_MAP'].lstrip("./\\"))
                if os.path.exists(alt_path):
                    region_config['DSM_MAP'] = alt_path
            # 如有需要也调整元数据 JSON 路径
            metadata_path = os.path.join('UAV_AVL_demo', 'Data', 'metadata')
            if os.path.isdir(metadata_path):
                config_metadata_file = os.path.join(metadata_path, f"{region}.json")
                if os.path.exists(config_metadata_file):
                    json_file = config_metadata_file
                else:
                    json_file = f'./Data/metadata/{region}.json'
            else:
                json_file = f'./Data/metadata/{region}.json'
        else:
            # 默认数据路径
            json_file = f'./Data/metadata/{region}.json'
        # # 从配置获取 UAV 图像路径和地点列表
        # UAV_path = find_values(region_config, 'UAV_PATH')
        # places = find_values(region_config, 'PLACES')
        # # 收集所有 UAV 图像文件路径
        # UAV_img_list0 = []
        # for place in places:
        #     base_path = UAV_path if UAV_path.endswith(os.sep) or UAV_path.endswith('/') else UAV_path + os.sep
        #     UAV_img_list0 += get_jpg_files(base_path + place)
            
        # # 根据测试步长选择子集图像（TEST_INTERVAL 控制采样间隔）
        # test_interval = region_config.get('TEST_INTERVAL', 1)
        # UAV_img_list = UAV_img_list0[0::test_interval]
        # # 方法字典初始化
        # method_dict = {}
        # # 遍历每个检索方法
        # for retrieval_method in All_Retrieval:
        #     method_dict['retrieval_method'] = retrieval_method
        #     method_dict = retrieval_init(method_dict, region_config)  # 初始化检索模型等
        #     # 遍历每个匹配方法
        #     for matching_method in All_Matching:
        #         method_dict['matching_method'] = matching_method
        #         method_dict = matching_init(method_dict)  # 初始化匹配方法
        #         # 加载该区域的参考地图和 DSM 数据（根据 Ref_type）
        #         ref_map0, dsm_map0, save_path0, ref_resolution = load_config_parameters_new(region_config, opt, region)
        #         if ref_map0 is None or dsm_map0 is None:
        #             raise FileNotFoundError(
        #                 f"Reference map or DSM not found for region {region}. 检查配置文件路径是否正确.")
        #         # 遍历该区域每张 UAV 图像
        #         for index, uav_path in enumerate(tqdm(UAV_img_list, desc=f"{region}", unit="image")):
        #             place = os.path.basename(os.path.dirname(uav_path))
        #             print(
        #                 f"Region: {region} | Place: {place} | Image: {os.path.basename(uav_path)} | Progress: {index / len(UAV_img_list) * 100:.1f}%")
        #             # 当前图像结果数据的保存路径（pickle 文件）
        #             VG_pkl_path = ('{}/{}/pkl_{}/resize_{}/{}-{}-{}-{}/VG_data_{}.pkl'
        #                            .format(opt.save_dir, region, place, opt.resize_ratio, opt.Ref_type,
        #                                    retrieval_method, matching_method, opt.pose_priori, img_name(uav_path)))
        #             # 若结果已存在则跳过
        #             if os.path.exists(VG_pkl_path):
        #                 continue
        #             # 中间结果保存目录
        #             save_path = f"{save_path0}/{place}/{index + 1}"
        #             # 读取真值位姿（元数据）
        #             truePos_list = query_data_from_file(json_file,
        #                                                 name=f"{os.path.dirname(uav_path)}/{os.path.basename(uav_path)}")
        #             if not truePos_list:
        #                 print(f"Warning: Metadata for image {uav_path} not found in {json_file}. Skipping.")
        #                 continue
        #             truePos = truePos_list[0]
        #             # 计算相机内参矩阵 K
        #             K = computeCameraMatrix(truePos)
        #             # 读取 UAV 图像
        #             uav_image = cv2.imread(uav_path)
        #             if uav_image is None:
        #                 print(f"Warning: Unable to read UAV image at {uav_path}. Skipping this image.")
        #                 continue
        #             # 根据 UAV 偏航角旋转参考地图
        #             ref_map, matRotation = dumpRotateImage(ref_map0, truePos['yaw'])
        #             # 记录该图像处理的起始时间
        #             VG_time0 = time.time()
        #             # 步骤1: 图像级检索，获取粗定位结果
        #             IR_order, refLocX, refLocY, PDE_list, cut_H, cut_W, fineScale, retrieval_time = retrieval_all(
        #                 ref_map, uav_path, truePos, ref_resolution, matRotation,
        #                 save_path, opt, region, region_config, method_dict)
        #             # 步骤2 & 3: 像素级匹配和 PnP 求解 UAV 位姿
        #             BLH_list, inliers_list, match_time, pnp_time = Match2Pos_all(
        #                 opt, region, region_config, uav_image, fineScale, K, ref_map, dsm_map0,
        #                 refLocY, refLocX, cut_H, cut_W, save_path, method_dict, matRotation)
        #             # 计算定位误差（预测位置与真值位置的距离）
        #             pred_loc, pred_error, location_error_list = pos2error(truePos, BLH_list, inliers_list)
        #             print(f"pred_error: {pred_error}")
        #             # 计算该图像总耗时
        #             VG_time_cost = time.time() - VG_time0
        #             # 保存所有相关数据到 pickle 以供后续分析
        #             save_data(
        #                 VG_pkl_path, opt=opt, region_config=region_config, img_path=uav_path,
        #                 truePos=truePos, refLocX=refLocX, refLocY=refLocY, IR_order=IR_order, PDE=PDE_list,
        #                 inliers=inliers_list, BLH_list=BLH_list, location_error_list=location_error_list,
        #                 pred_loc=pred_loc, pred_error=pred_error, retrieval_time=retrieval_time,
        #                 match_time=match_time, pnp_time=pnp_time, total_time=VG_time_cost
        #             )



        # ================== 25：从配置获取 UAV 图像路径和地点列表 ==================
        UAV_path = find_values(region_config, 'UAV_PATH')
        places = find_values(region_config, 'PLACES')
        if isinstance(UAV_path, (list, tuple)):
            UAV_path = UAV_path[0]

        # ================== 收集所有 UAV 图像完整路径（逐 place，自然序） ==================
        import os, re
        def _natkey(p):
            s = os.path.basename(p)
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

        UAV_img_list0 = []
        base_path = UAV_path if str(UAV_path).endswith(('/', os.sep)) else str(UAV_path) + os.sep
        # 保持 places 的原有顺序；目录内自然序
        for place in places:
            jpgs = get_jpg_files(base_path + place)  # 扫描出每个目录内的完整路径
            jpgs = sorted(jpgs, key=_natkey)
            UAV_img_list0.extend(jpgs)

        # 默认的测试间隔（若给了固定清单，后面会置为 1）
        test_interval = region_config.get('TEST_INTERVAL', config.get('TEST_INTERVAL', 1))

        # ================== 可选：从固定清单强制对齐（从 config 读取，而不是 opt） ==================
        list_path = config.get('UAV_IMAGE_LIST_FILE', None)  # YAML 顶层键，已确认存在:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
        if list_path:
            list_path = os.path.abspath(os.path.expanduser(str(list_path)))
            print(f"[INFO] UAV_IMAGE_LIST_FILE => {list_path}")
        if list_path and os.path.isfile(list_path):
            with open(list_path, 'r', encoding='utf-8') as f:
                raw_lines = [ln.strip().lstrip('\ufeff') for ln in f if ln.strip()]
            print(f"[INFO] Read {len(raw_lines)} entries from manifest.")

            # 建大小写不敏感映射：basename(lower) -> fullpath（来自上面扫描的 UAV_img_list0）
            name2path_ci = {os.path.basename(p).lower(): p for p in UAV_img_list0}

            fixed_paths, missing = [], []
            for n in raw_lines:
                key = os.path.basename(n).lower()  # 只比对 Basename
                hit = name2path_ci.get(key, None)
                if hit: fixed_paths.append(hit)
                else:   missing.append(os.path.basename(n))

            if missing:
                print(f"[WARN] {len(missing)} items in UAV_IMAGE_LIST_FILE not found in data, e.g. {missing[:2]}")

            if fixed_paths:
                UAV_img_list0 = fixed_paths
                test_interval = 1  # 清单驱动时不再做间隔采样
            else:
                print("[WARN] Manifest resolved to 0 images; fallback to scanned list.")
        else:
            if list_path:
                print(f"[WARN] UAV_IMAGE_LIST_FILE not found: {list_path}")

        # ================== 采样（间隔） ==================
        UAV_img_list = UAV_img_list0[0::test_interval]

        # ================== 将本次用于评测的清单（仅文件名）落盘便于核对 ==================
        os.makedirs(opt.save_dir, exist_ok=True)
        used_names_txt = os.path.join(opt.save_dir, f"{region}_uav_images_used.txt")
        with open(used_names_txt, "w", encoding="utf-8") as f:
            f.write("\n".join([os.path.basename(p) for p in UAV_img_list]))
        print(f"[INFO] Wrote list of used UAV images to {used_names_txt}")

        # ================== 方法字典初始化（只取第一项；修复 NameError） ==================
        method_dict = {}

        # 读取 YAML 中的方法列表（在你程序前面读取的 All_Retrieval / All_Matching）
        try:
            print(f"[DBG] All_Retrieval: {All_Retrieval}")
        except Exception:
            All_Retrieval = find_values(config, 'RETRIEVAL_METHODS') or []
        try:
            print(f"[DBG] All_Matching: {All_Matching}")
        except Exception:
            All_Matching = find_values(config, 'MATCHING_METHODS') or []

        if not All_Retrieval:
            raise RuntimeError("No retrieval methods found in config (RETRIEVAL_METHODS).")
        if not All_Matching:
            raise RuntimeError("No matching methods found in config (MATCHING_METHODS).")

        retrieval_method = All_Retrieval[0]
        matching_method  = All_Matching[0]
        print(f"[DBG] >>> init retrieval: {retrieval_method}")
        method_dict['retrieval_method'] = retrieval_method
        method_dict = retrieval_init(method_dict, region_config)  # 初始化检索模型等

        print(f"[DBG] >>> init matching: {matching_method}")
        method_dict['matching_method'] = matching_method
        method_dict = matching_init(method_dict)  # 初始化匹配方法

                # Load the reference map and DSM for the current region
        ref_map0, dsm_map0, save_path0, ref_resolution = load_config_parameters_new(region_config, opt, region)
        if ref_map0 is None or dsm_map0 is None:
            raise FileNotFoundError(f"Reference map or DSM not found for region {region}. 检查配置文件路径是否正确.")

        # Iterate over each UAV image in this region
        for index, uav_path in enumerate(tqdm(UAV_img_list, desc=f"{region}", unit="image")):
            place = os.path.basename(os.path.dirname(uav_path))
            print(f"Region: {region} | Place: {place} | Image: {os.path.basename(uav_path)} | Progress: {index / len(UAV_img_list) * 100:.1f}%")

            # Prepare output pickle path for this image’s results
            VG_pkl_path = ('{}/{}/pkl_{}/resize_{}/{}-{}-{}-{}/VG_data_{}.pkl'
                        .format(opt.save_dir, region, place, opt.resize_ratio, opt.Ref_type,
                                retrieval_method, matching_method, opt.pose_priori, img_name(uav_path)))
            # Skip if result already exists
            if os.path.exists(VG_pkl_path):
                continue

            # Create directory for intermediate results of this image
            save_path = f"{save_path0}/{place}/{index + 1}"

            # Retrieve ground-truth pose (metadata) for the UAV image
            truePos_list = query_data_from_file(json_file, name=f"{os.path.dirname(uav_path)}/{os.path.basename(uav_path)}")
            if not truePos_list:
                print(f"Warning: Metadata for image {uav_path} not found in {json_file}. Skipping.")
                continue
            truePos = truePos_list[0]

            # Compute camera intrinsic matrix K from true pose
            K = computeCameraMatrix(truePos)
            # Read the UAV image
            uav_image = cv2.imread(uav_path)
            if uav_image is None:
                print(f"Warning: Unable to read UAV image at {uav_path}. Skipping this image.")
                continue

            # Rotate the reference map according to the UAV yaw angle
            ref_map, matRotation = dumpRotateImage(ref_map0, truePos['yaw'])
            # Start timer for this image’s processing
            VG_time0 = time.time()

            # Step 1: Image-level retrieval to get coarse localization
            IR_order, refLocX, refLocY, PDE_list, cut_H, cut_W, fineScale, retrieval_time = retrieval_all(
                ref_map, uav_path, truePos, ref_resolution, matRotation,
                save_path, opt, region, region_config, method_dict
            )

            # Step 2 & 3: Pixel-level matching and PnP to solve UAV pose
            BLH_list, inliers_list, match_time, pnp_time = Match2Pos_all(
                opt, region, region_config, uav_image, fineScale, K, ref_map, dsm_map0,
                refLocY, refLocX, cut_H, cut_W, save_path, method_dict, matRotation
            )

            # Compute localization error (distance between predicted and true position)
            pred_loc, pred_error, location_error_list = pos2error(truePos, BLH_list, inliers_list)
            print(f"pred_error: {pred_error}")

            # Calculate total processing time for this image
            VG_time_cost = time.time() - VG_time0

            # Save all results and data for this image to a pickle file
            save_data(
                VG_pkl_path, opt=opt, region_config=region_config, img_path=uav_path,
                truePos=truePos, refLocX=refLocX, refLocY=refLocY, IR_order=IR_order, PDE=PDE_list,
                inliers=inliers_list, BLH_list=BLH_list, location_error_list=location_error_list,
                pred_loc=pred_loc, pred_error=pred_error, retrieval_time=retrieval_time,
                match_time=match_time, pnp_time=pnp_time, total_time=VG_time_cost
            )


        # ===== 重要：不再在这里做任何 return/exit，也不再 for 循环。
        #      直接让 240 行之后你原来的主体代码继续执行（加载参考地图、逐图处理、PnP 等）。


