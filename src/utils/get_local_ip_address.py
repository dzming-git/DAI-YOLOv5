def get_local_ip_address():
    ip_address = ''
    try:
        # 获取IP地址
        import subprocess
        ip_address = subprocess.check_output(['hostname', '-I']).decode('utf-8').strip()
    except:
        pass
    return ip_address
