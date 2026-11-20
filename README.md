# 智能交易台

## 部署到 Railway

1. 注册 https://railway.app
2. 新建项目 → Deploy from GitHub（或直接上传）
3. 设置环境变量（可选）：
   - PORT: 自动
4. 部署后访问生成的域名

## 本地运行

```bash
pip install -r requirements.txt
python smart_paper_trader.py
```

访问 http://127.0.0.1:5000

## 配置

- 每台策略独立 100U
- 手续费 0.1%
- 更新间隔 60 秒
