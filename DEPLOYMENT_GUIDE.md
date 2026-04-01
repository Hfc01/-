# 部署指南：将情感分析系统部署到 hfctree.top

本指南将帮助您将基于Streamlit的情感分析系统部署到您的网站 hfctree.top 上。

## 准备工作

### 1. 服务器环境准备

假设您已经有一台运行Linux的服务器，并且已经配置了域名 hfctree.top 指向该服务器。

### 2. 安装必要的软件

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和相关工具
sudo apt install python3 python3-pip python3-venv nginx supervisor git -y
```

## 部署步骤

### 1. 克隆项目代码

```bash
# 进入网站根目录
cd /var/www

# 克隆项目代码
git clone <您的项目仓库地址> sentiment-analysis

# 进入项目目录
cd sentiment-analysis
```

### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置Streamlit

创建Streamlit配置文件：

```bash
mkdir -p ~/.streamlit

cat > ~/.streamlit/config.toml << EOF
[server]
port = 8501
enableCORS = false
headless = true
EOF
```

### 4. 配置Supervisor

创建Supervisor配置文件，用于管理Streamlit进程：

```bash
sudo nano /etc/supervisor/conf.d/sentiment-analysis.conf
```

添加以下内容：

```ini
[program:sentiment-analysis]
directory=/var/www/sentiment-analysis
command=/var/www/sentiment-analysis/venv/bin/streamlit run model/web_demo.py
autostart=true
autorestart=true
startsecs=10
user=www-data
redirect_stderr=true
stdout_logfile=/var/log/supervisor/sentiment-analysis.log
```

### 5. 配置Nginx

创建Nginx配置文件：

```bash
sudo nano /etc/nginx/sites-available/sentiment-analysis
```

添加以下内容（将 hfctree.top 替换为您的实际域名）：

```nginx
server {
    listen 80;
    server_name hfctree.top;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/sentiment-analysis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. 启动服务

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start sentiment-analysis
```

### 7. 配置SSL（可选）

如果您需要HTTPS，可以使用Let's Encrypt：

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d hfctree.top
```

## 维护与管理

### 查看服务状态

```bash
sudo supervisorctl status sentiment-analysis
```

### 查看日志

```bash
sudo tail -f /var/log/supervisor/sentiment-analysis.log
```

### 重启服务

```bash
sudo supervisorctl restart sentiment-analysis
```

### 更新代码

```bash
cd /var/www/sentiment-analysis
git pull
sudo supervisorctl restart sentiment-analysis
```

## 故障排除

1. **Streamlit服务无法启动**：检查日志文件，确认依赖是否正确安装
2. **Nginx 404错误**：确认Nginx配置文件是否正确，Streamlit服务是否运行
3. **权限问题**：确保www-data用户有访问项目目录的权限

## 注意事项

- 确保服务器有足够的内存（建议至少2GB）
- 定期更新依赖包以确保安全性
- 考虑设置防火墙规则，只允许必要的端口访问

部署完成后，您应该可以通过 https://hfctree.top 访问您的情感分析系统。