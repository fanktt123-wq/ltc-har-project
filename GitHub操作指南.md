# GitHub 操作指南

本指南将帮助您完成以下任务：
1. 创建私有GitHub仓库
2. 添加团队成员为开发人员
3. 提交代码到GitHub
4. 在SEP系统中提交GitHub链接和组员分工名单

---

## 一、创建私有GitHub仓库

### 步骤1：登录GitHub

1. 访问 [GitHub官网](https://github.com)
2. 使用您的GitHub账号登录（如果没有账号，请先注册）

### 步骤2：创建新仓库

1. 点击页面右上角的 **"+"** 按钮，选择 **"New repository"**（新建仓库）
2. 填写仓库信息：
   - **Repository name（仓库名称）**：例如 `ltc-har-project` 或 `lnn-human-activity-recognition`
   - **Description（描述）**：可选，例如 "基于LTC的液态神经网络在UCI HAR人体活动识别中的应用"
   - **Visibility（可见性）**：选择 **"Private"（私有）** ⚠️ **重要：建议选择私有**（原因见下方说明）
   - **Initialize this repository with**（初始化选项）：
     - ✅ 可以勾选 "Add a README file"（如果已经创建了README.md，可以不勾选）
     - ❌ 不要勾选 "Add .gitignore"（除非需要）
     - ❌ 不要勾选 "Choose a license"（除非需要）
3. 点击 **"Create repository"**（创建仓库）按钮

### 为什么选择私有仓库？

**私有仓库（Private）的优势：**
- ✅ **保护学术成果**：防止代码被他人直接复制，保护团队的研究成果
- ✅ **符合学术规范**：课程作业通常要求私有，避免抄袭问题
- ✅ **控制访问权限**：只有被邀请的成员才能查看和访问代码
- ✅ **灵活管理**：可以随时将私有仓库改为公开，但公开仓库不能改为私有（免费账户）
- ✅ **避免误用**：防止他人直接使用你的代码而不注明来源

**公有仓库（Public）的特点：**
- ⚠️ 任何人都可以查看、复制、fork你的代码
- ⚠️ 适合开源项目，但不适合课程作业
- ⚠️ 如果作业要求私有，使用公有仓库可能不符合要求

**注意：** 
- 如果SEP系统明确要求使用私有仓库，则必须选择Private
- 如果只是建议，也可以根据实际情况选择，但私有更安全
- 私有仓库在GitHub免费账户中完全可用，无额外费用

---

## 二、将本地项目推送到GitHub

### 步骤1：初始化Git仓库（如果还没有）

打开终端（PowerShell或命令提示符），进入项目目录：

```bash
cd C:\Users\83851\Documents\lnn
```

如果项目还没有初始化Git，执行：

```bash
git init
```

### 步骤2：添加文件到Git

```bash
# 添加所有文件
git add .

# 或者只添加特定文件
git add README.md
git add *.py
git add config.py
```

### 步骤3：创建初始提交

```bash
git commit -m "Initial commit: LTC HAR project"
```

### 步骤4：连接到GitHub远程仓库

在GitHub仓库页面，复制仓库的HTTPS或SSH地址，然后执行：

```bash
# 使用HTTPS（推荐，更简单）
git remote add origin https://github.com/你的用户名/仓库名.git

# 例如：
# git remote add origin https://github.com/zhangsan/ltc-har-project.git
```

### 步骤5：推送代码到GitHub

```bash
# 推送主分支（main或master）
git branch -M main
git push -u origin main
```

**注意：** 首次推送可能需要输入GitHub用户名和密码（或Personal Access Token）

---

## 三、添加团队成员为开发人员

### 步骤1：进入仓库设置

1. 在GitHub仓库页面，点击 **"Settings"（设置）** 标签
2. 在左侧菜单中找到 **"Collaborators"（协作者）** 或 **"Manage access"（管理访问权限）**

### 步骤2：添加协作者

1. 点击 **"Add people"（添加人员）** 或 **"Invite a collaborator"（邀请协作者）** 按钮
2. 输入组员的GitHub用户名或邮箱地址
3. 选择权限级别：
   - **Write（写入）**：允许推送代码、创建分支等（推荐选择此权限）
   - **Admin（管理员）**：拥有所有权限（通常只有组长需要）
4. 点击 **"Add [username] to this repository"（添加用户到仓库）**

### 步骤3：组员接受邀请

1. 组员会收到GitHub的邮件邀请
2. 组员登录GitHub，点击邮件中的链接接受邀请
3. 或者组员在GitHub首页的 **"Invitations"（邀请）** 中接受邀请

---

## 四、组员克隆仓库并开始协作

### 步骤1：组员克隆仓库

组员在本地执行：

```bash
git clone https://github.com/你的用户名/仓库名.git
cd 仓库名
```

### 步骤2：创建分支（推荐工作流程）

```bash
# 创建并切换到新分支
git checkout -b feature/成员姓名-功能描述

# 例如：
# git checkout -b feature/zhangsan-add-visualization
```

### 步骤3：提交更改

```bash
# 添加修改的文件
git add .

# 提交更改
git commit -m "描述你的更改"

# 推送到远程仓库
git push origin feature/成员姓名-功能描述
```

### 步骤4：创建Pull Request（可选）

1. 在GitHub仓库页面，点击 **"Pull requests"（拉取请求）**
2. 点击 **"New pull request"（新建拉取请求）**
3. 选择你的分支，填写描述
4. 点击 **"Create pull request"（创建拉取请求）**
5. 组长审查后合并到主分支

---

## 五、在SEP系统中提交

### 需要准备的内容

1. **GitHub仓库链接**：
   - 格式：`https://github.com/你的用户名/仓库名`
   - 例如：`https://github.com/zhangsan/ltc-har-project`

2. **组员分工名单**：
   建议格式如下：

   ```
   项目名称：基于LTC的液态神经网络在UCI HAR人体活动识别中的应用
   
   团队成员及分工：
   
   1. [组长姓名] - 学号：[学号]
      - 负责：项目整体架构设计、模型实现、训练流程
      - GitHub用户名：[GitHub用户名]
   
   2. [成员1姓名] - 学号：[学号]
      - 负责：数据处理、数据集加载模块
      - GitHub用户名：[GitHub用户名]
   
   3. [成员2姓名] - 学号：[学号]
      - 负责：评估指标计算、可视化模块
      - GitHub用户名：[GitHub用户名]
   
   4. [成员3姓名] - 学号：[学号]
      - 负责：实验报告撰写、结果分析
      - GitHub用户名：[GitHub用户名]
   ```

### 提交步骤

1. 登录SEP系统
2. 找到相应的作业/项目提交入口
3. 填写以下信息：
   - **GitHub仓库链接**：粘贴完整的仓库URL
   - **组员分工说明**：粘贴上述格式的分工名单
4. 提交

---

## 六、常见问题解决

### 问题1：推送时提示需要认证

**解决方案：**
- 使用Personal Access Token（个人访问令牌）代替密码
- 生成Token：GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
- 权限选择：至少勾选 `repo` 权限

### 问题2：无法推送代码

**解决方案：**
```bash
# 检查远程仓库配置
git remote -v

# 如果地址不对，重新设置
git remote set-url origin https://github.com/你的用户名/仓库名.git
```

### 问题3：组员看不到仓库

**解决方案：**
- 确认组员已接受邀请
- 确认仓库是私有仓库（Private）
- 确认组员有正确的权限（Write或Admin）

### 问题4：合并冲突

**解决方案：**
```bash
# 拉取最新代码
git pull origin main

# 如果有冲突，解决冲突后
git add .
git commit -m "解决合并冲突"
git push origin main
```

---

## 七、Git基本命令速查

```bash
# 查看状态
git status

# 查看提交历史
git log

# 查看分支
git branch

# 切换分支
git checkout 分支名

# 拉取最新代码
git pull origin main

# 查看远程仓库
git remote -v
```

---

## 八、推荐工作流程

1. **组长**：
   - 创建私有仓库
   - 推送初始代码
   - 添加组员为协作者
   - 在SEP系统提交GitHub链接和分工名单

2. **组员**：
   - 接受邀请
   - 克隆仓库
   - 创建自己的分支进行开发
   - 定期推送代码

3. **协作**：
   - 使用分支进行功能开发
   - 通过Pull Request进行代码审查
   - 定期合并到主分支

---

## 需要帮助？

- GitHub官方文档：https://docs.github.com
- Git官方文档：https://git-scm.com/doc
- 如果遇到问题，可以查看GitHub仓库的Issues或联系技术支持

---

**祝您使用愉快！** 🎉

