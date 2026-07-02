import os
import logging
from glob import glob

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_eye_folders(root_path):
    """
    统计根目录下所有患者的OD和OS眼睛数据文件夹数量

    参数:
        root_path: 包含所有患者文件夹的根目录路径

    返回:
        一个字典，包含总患者数、OD眼数量、OS眼数量和总数
    """
    # 初始化计数器
    total_patients = 0
    od_count = 0
    os_count = 0

    # 遍历根目录下的所有患者文件夹
    for item in os.listdir(root_path):
        patient_path = os.path.join(root_path, item)

        # 只处理文件夹
        if not os.path.isdir(patient_path):
            continue

        total_patients += 1
        patient_has_od = False
        patient_has_os = False

        # 检查该患者是否有OD相关文件夹
        # 搜索所有包含"OD"的子文件夹
        od_patterns = [
            os.path.join(patient_path, '**', '*OD*'),
            os.path.join(patient_path, '*OD*')
        ]

        for pattern in od_patterns:
            od_folders = glob(pattern, recursive=True)
            if any(os.path.isdir(folder) for folder in od_folders):
                patient_has_od = True
                break

        # 检查该患者是否有OS相关文件夹
        os_patterns = [
            os.path.join(patient_path, '**', '*OS*'),
            os.path.join(patient_path, '*OS*')
        ]

        for pattern in os_patterns:
            os_folders = glob(pattern, recursive=True)
            if any(os.path.isdir(folder) for folder in os_folders):
                patient_has_os = True
                break

        # 更新计数器
        if patient_has_od:
            od_count += 1
            logger.debug(f"患者 {item} 有OD(右眼)数据")
        if patient_has_os:
            os_count += 1
            logger.debug(f"患者 {item} 有OS(左眼)数据")

    # 汇总结果
    result = {
        'total_patients': total_patients,
        'od_eye_count': od_count,
        'os_eye_count': os_count,
        'total_eye_count': od_count + os_count
    }

    return result


def main():
    # 请在这里修改为你的数据根目录路径
    root_data_path = "/data/yzr/nas/home/SCL/"  # 用户需要修改这个路径

    if not os.path.exists(root_data_path):
        logger.error(f"路径不存在: {root_data_path}")
        return

    logger.info(f"开始统计 {root_data_path} 下的眼睛数据文件夹...")
    stats = count_eye_folders(root_data_path)

    # 打印统计结果
    print("\n===== 眼睛数据统计结果 =====")
    print(f"总患者数: {stats['total_patients']}")
    print(f"OD(右眼)数量: {stats['od_eye_count']}")
    print(f"OS(左眼)数量: {stats['os_eye_count']}")
    print(f"总眼睛数据数量: {stats['total_eye_count']}")
    print("===========================\n")


if __name__ == "__main__":
    main()
