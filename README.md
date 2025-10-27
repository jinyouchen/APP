# ML DevOps App 项目说明
## 1. 流程说明（commit→build→test→staging→prod）
- **commit**：开发者在feature分支开发，提交代码到Git
- **build**：PR到dev/staging/main时，CI自动构建Docker镜像（符合1-20）
- **test**：CI自动运行测试套件+Lint（符合1-18至1-19）
- **staging**：PR合并到staging后，用staging Secrets生成.env，构建staging镜像（符合1-28）
- **prod**：PR合并到main后，用prod Secrets生成.env，构建prod镜像（符合1-29）

## 2. 分支触发规则（符合1-35）
- **feature分支**：开发分支，合并到dev触发CI
- **dev分支**：开发主分支，PR到staging触发CI；合并到staging触发staging CD
- **staging分支**：预发布分支，PR到main触发CI；合并到main触发prod CD
- **main分支**：生产分支，仅接受staging的合并PR

## 3. 核心依赖
- Git、Docker、GitHub Actions（DevOps）
- Python、scikit-learn、MLflow（MLOps）