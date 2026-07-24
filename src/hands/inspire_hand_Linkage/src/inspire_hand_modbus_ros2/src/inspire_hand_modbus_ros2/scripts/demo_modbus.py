from pymodbus.client import ModbusTcpClient  
import time

# 寄存器字典
regdict = {
    'ID': 1000,
    'baudrate': 1001,
    'clearErr': 1004,
    'forceClb': 1009,
    'angleSet': 1486,
    'forceSet': 1498,
    'speedSet': 1522,
    'angleAct': 1546,
    'forceAct': 1582,
    'Action': 1600, 
    'errCode': 1606,
    'statusCode': 1612,
    'temp': 1618,
    'actionSeq': 2320,
    'actionRun': 2322
}

def open_modbus(ip, port):
    client = ModbusTcpClient(host=ip, port=port)
    if not client.connect():
        print('无法连接到 Modbus 服务器')
        return None
    print('成功连接到 Modbus 服务器')
    return client

def write_register(client, address, values):
    response = client.write_registers(address, values)
    if response.isError():
        print('写入寄存器出错')

def read_register(client, address, count=1):
    response = client.read_holding_registers(address=address, count=count)
    if response.isError():
        print('读取寄存器出错')
        return []
    return response.registers

def write6(client, reg_name, val):
    if reg_name in ['angleSet', 'forceSet', 'speedSet']:
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFFFF)  # 取低16位
        write_register(client, regdict[reg_name], val_reg)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')

def read6(client, reg_name):
    if reg_name in ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct', 'Action']:
        val = read_register(client, regdict[reg_name], 6)
        if len(val) < 6:
            print('没有读到数据')
            return
        print('读到的值依次为：', end='')
        for v in val:
            print(v, end=' ')
        print()
    
    elif reg_name in ['errCode', 'statusCode', 'temp']:
        val_act = read_register(client, regdict[reg_name], 3)
        if len(val_act) < 3:
            print('没有读到数据')
            return
            
        results = []
        for i in range(len(val_act)):
            low_byte = val_act[i] & 0xFF            
            high_byte = (val_act[i] >> 8) & 0xFF    
            results.append(low_byte)  
            results.append(high_byte)  

        print('读到的值依次为：', end='')
        for v in results:
            print(v, end=' ')
        print()
    
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'temp\'')

if __name__ == '__main__':
    ip_address = '192.168.11.210'
    port = 6000
    print('打开Modbus TCP连接！')
    client = open_modbus(ip_address, port)
    
    if client is None:
        print('连接失败，退出程序')
        exit(1)

    print('设置灵巧手运动速度参数，-1为不设置该运动速度！')
    write6(client, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
    time.sleep(1)
    
    print('设置灵巧手抓握力度参数！')
    write6(client, 'forceSet', [500, 500, 500, 500, 500, 500])
    time.sleep(1)
    
    print('设置灵巧手运动角度参数0，-1为不设置该运动角度！')
    write6(client, 'angleSet', [0, 0, 0, 0, 400, -1])
    time.sleep(1)
    
    read6(client, 'angleAct')
    time.sleep(1)
    
    print('设置灵巧手运动角度参数1000，-1为不设置该运动角度！')
    write6(client, 'angleSet', [1000, 1000, 1000, 1000, 1000, -1])
    time.sleep(1)
    
    read6(client, 'angleAct')
    time.sleep(1)
    
    print('故障信息：')
    read6(client, 'errCode')
    time.sleep(1)
    
    print('电缸温度：')
    read6(client, 'temp')
    time.sleep(1)
    
    print('设置灵巧手动作库序列：2！')
    write_register(client, regdict['actionSeq'], [2])
    time.sleep(1)
    
    print('运行灵巧手当前序列动作！')
    write_register(client, regdict['actionRun'], [1])
    
    # 关闭 Modbus TCP 连接
    client.close()
    print('关闭Modbus TCP连接！')

