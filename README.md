# Flight Reservation System

## Project Framework

- main    → Protected, only through GitHub Pull Request

- develop → Main Develop Branch

## Daily Workflow Instruction

### 一、同步正式 Main 內容

```bash
git checkout main
git pull
```

### 二、每次開發新功能

1. 先確保 develop 是最新的

```bash
git checkout develop
git merge main
git pull
```

2. 建立自己的功能分支

```bash
git checkout -b feature/你的功能名稱
```

3. 改完程式碼後

```bash
git add 檔名
git commit -m "說明做了什麼"
git push origin feature/你的功能名稱
```

4. 前往 GitHub 發佈 Pull Request

- base: `develop`
- compare: `feature/你的功能名稱`
- Waiting for Approve, then Merging

### 三、Merge 完後 Clear Branch

```bash
# 刪除本地分支
git branch -d feature/你的功能名稱

# 刪除 GitHub 上的分支
git push origin --delete feature/你的功能名稱
```

### 四、同步 develop（After Successfully Merged in Main）

```bash
git checkout develop
git merge main
git push origin develop
```
