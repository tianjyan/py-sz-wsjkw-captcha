健康苏州掌上行验证码识别

# 使用

验证码获取地址：https://app.wsjkw.suzhou.com.cn/VerifyImage?date=1688957732978

其中date是时间戳，可以随意生成。

```bash
python -m pip install -r requirements.txt // 安装依赖
python prepare.py                         // 预处理数据
python train.py                           // 训练模型
python work.pyt < test.jpg                // 验证模型准确性

```