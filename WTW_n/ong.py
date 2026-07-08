from PIL import Image
import os


def convert_tif_to_png(input_path, output_path=None):
    """
    将 TIF 图片转换为 PNG 格式
    :param input_path: 输入的 TIF 文件路径
    :param output_path: 输出的 PNG 文件路径（可选，默认自动生成）
    :return: 输出文件路径
    """
    # 打开 TIF 文件
    with Image.open(input_path) as img:
        # 如果未指定输出路径，则自动生成（替换扩展名）
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".png"

        # 转换为 PNG 并保存
        img.save(output_path, "PNG")
        print(f"转换成功！文件已保存至: {output_path}")
        return output_path


# 示例用法
input_file = "D:\deeplearning\WTW\ZJD_OS_B_img001.tif"  # 替换为你的 TIF 文件路径
convert_tif_to_png(input_file)