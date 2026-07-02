import numpy as np
from PIL import Image

from DataSet import data_SCL

datasets_map = {
   'SCL': data_SCL
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size, img_width, seq_length, mode='train'):
    if dataset_name not in datasets_map:
        raise ValueError('未能在datamap找到，请检查dataname！')

    if dataset_name == 'SCL':
        input_param = {
            'paths': train_data_paths,
            'image_width': img_width,
            'minibatch_size': batch_size,
            'seq_length': seq_length,
            'input_data_type': 'float32',
            'name': dataset_name + 'iterator',
            'mode': mode
        }
        input_handle_generator = datasets_map[dataset_name].DataProcess(input_param)

        if mode == 'train':
            train_input_handle = input_handle_generator.get_train_input_handle()
            train_input_handle.begin(do_shuffle=False, shuffle_within_group=True)
            return train_input_handle

        elif mode == 'valid':
            valid_input_handle = input_handle_generator.get_valid_input_handle()
            valid_input_handle.begin(do_shuffle=False, shuffle_within_group=False)
            return valid_input_handle

        elif mode == 'test':
            test_input_handle = input_handle_generator.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False, shuffle_within_group=False)
            return test_input_handle
        else:
            raise ValueError(f"不支持的模式: {mode}")

if __name__ == '__main__':
    DATA_PATH = r"/public/home/yuanzr/project/CLFitNet_ICL/SCL"
    SEQUENCE_LENGTH = 3
    BATCH_SIZE = 1
    IMAGE_WIDTH = 512
    MODE = 'test'

    print("开始测试数据加载流程...")

    test_input_handle = data_provider(
        dataset_name='SCL',
        train_data_paths=DATA_PATH,
        valid_data_paths=DATA_PATH,
        batch_size=BATCH_SIZE,
        img_width=IMAGE_WIDTH,
        seq_length=SEQUENCE_LENGTH,
        mode=MODE
    )

    if test_input_handle and test_input_handle.total() > 0:
        print(f"\n成功初始化 '{MODE}' 数据迭代器。总共有 {test_input_handle.total()} 个序列。")
        ims, csjs = test_input_handle.get_batch() # 这里进行解包

        if ims is not None:
            print(f"成功获取一个批次的数据，形状为: {ims.shape}, CSJ形状为: {csjs.shape}")
            real_ims = ims[:, :, :, :, :1]

            print("正在显示批次中第一个序列的图像及拱高...")
            for i in range(SEQUENCE_LENGTH):
                img_gt = np.uint8(real_ims[0, i, :, :, :] * 255)
                img_gt = np.reshape(img_gt, img_gt.shape[:-1])
                img_gt_img = Image.fromarray(img_gt)
                
                time_point_map = {0: "initial", 1: "30mins", 2: "1h", 3: "2h", 4: "4h"}
                print(f"已显示序列中的第 {i + 1} 张图像: {time_point_map.get(i, f'时间点{i+1}')} | CSJ_mm: {csjs[0, i]}")
        else:
            print("获取批次失败。")
    else:
        print("\n未能获取到数据。请检查路径和结构")