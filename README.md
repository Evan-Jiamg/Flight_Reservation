# Flight Reservation System

## Project Framework

- main    → 受保護，只能透過 PR 進入
- develop → 主要開發分支

## Daily Workflow Instruction:

### 每次開發新功能: 

1. 先確保 develop 是最新的

```
git checkout develop
git pull
```

2. 建立自己的功能分支

```
git checkout -b feature/你的功能名稱
```

3. 改完程式碼後
```
git add 檔名
git commit -m "說明做了什麼"
git push origin feature/你的功能名稱
```