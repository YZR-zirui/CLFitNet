import os

# 把路径一行写清楚
check_path = r"/data/yzr/nas/home/SCL_all/"

print(f"测试路径: [{check_path}]")

if os.path.isdir(check_path):
    print("✅ 目录存在！")
else:
    print("❌目录不存在")
