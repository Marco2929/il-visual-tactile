import time
from pymodbus.client import ModbusTcpClient

# 定义 Modbus TCP 相关参数
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000
TOUCH_SENSOR_BASE_ADDR = 3000

def read_single_register():
    client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
    client.connect()

    try:
        while True:
            # 读取 3000地址的寄存器值
            response = client.read_holding_registers(TOUCH_SENSOR_BASE_ADDR, 1)

            if response.isError():
                print("读取寄存器失败:", response)
            else:
                register_value = response.registers[0]  # 获取寄存器值
                print(f"寄存器 3032 的值: {register_value}")

            time.sleep(0.001)  

    finally:
        client.close()  # 退出时关闭连接

if __name__ == "__main__":
    read_single_register()

