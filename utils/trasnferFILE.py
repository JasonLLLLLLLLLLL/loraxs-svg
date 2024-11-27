import os
import paramiko


def transfer_files(target_folder, host_ip, remote_target_dir, username, password):
    # 创建一个SSH客户端
    ssh = paramiko.SSHClient()
    # 自动添加目标服务器的SSH密钥（注意这不太安全）
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 连接到远程服务器
        ssh.connect(host_ip, username=username, password=password)

        # 使用SFTP进行文件传输
        sftp = ssh.open_sftp()

        # 遍历本地文件夹中的所有文件
        for file_name in os.listdir(target_folder):
            local_file_path = os.path.join(target_folder, file_name)
            if os.path.isfile(local_file_path):  # 只上传文件
                remote_file_path = os.path.join(remote_target_dir, file_name)
                sftp.put(local_file_path, remote_file_path)
                print(f"已传输 {file_name} 到 {remote_file_path}")

        # 关闭SFTP连接
        sftp.close()

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 关闭SSH连接
        ssh.close()


# 设置参数
target_folder = '/home/liuzhe/new-files/result/32-origin'  # 本地文件夹路径
host_ip = '10.198.28.57'
remote_target_dir = '/home/lz/SSD-GreatWall/newERA/xs-result/32-path'  # 远程服务器上的目标目录
username = 'lz'
password = '12wq'

# 执行文件传输
transfer_files(target_folder, host_ip, remote_target_dir, username, password)