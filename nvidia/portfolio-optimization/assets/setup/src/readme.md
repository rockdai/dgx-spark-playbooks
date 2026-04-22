# GPU 加速的产品组合优化

NVIDIA 定量投资组合优化开发人员示例使用 NVIDIA cuOpt 和 CUDA-X 数据科学库将投资组合优化从缓慢的批处理流程转变为快速的迭代工作流程。  GPU 加速的投资组合优化管道可实现可扩展的策略回测和交互式分析。

## 概述

该软件包提供了一整套用于量化投资组合管理的工具，包括风险意识优化、回溯测试和动态再平衡。该库通过 NVIDIA 的 cuOpt 求解器利用 GPU 加速来高效处理大规模投资组合优化问题。

## 主要特点

- **GPU 加速的 CVaR 优化**：利用 NVIDIA cuOpt 进行快速、可扩展的产品组合优化
- **多种建模 API**：与 CVXPY（CPU 和 GPU）和 cuOpt Python API（GPU）兼容
- **高级风险管理**：基于 CVaR 的下行风险控制，具有可定制的约束
- **动态再平衡**：具有可配置触发条件的系统性投资组合再平衡
- **全面回测**：针对具有多个指标的基准进行性能评估
- **场景生成**：使用几何布朗运动生成合成数据
- **灵活的约束**：权重限制、杠杆限制、营业额限制、基数限制

## 模块结构

### 核心优化

#### `cvar_optimizer.CVaR`
主要 CVaR 投资组合优化器类支持具有多个求解器接口的 Mean-CVaR 优化。

**关键能力：**
- CVXPY 求解器集成（CPU）
- cuOpt 求解器集成 (GPU)
- 可定制的约束框架
- 支持权重限制、杠杆限制、CVaR 硬限制、周转限制和基数限制

#### `base_optimizer.BaseOptimizer`
抽象基类为优化算法提供通用功能，包括权重约束处理和投资组合状态管理。

#### `cvar_parameters.CvarParameters`
CVaR 优化参数、约束和求解器设置的配置类。

#### `cvar_data.CvarData`
用于返回场景、资产信息和优化输入的数据容器。

#### `cvar_utils`
用于 CVaR 计算、投资组合评估和可视化以及优化求解器基准辅助方法的实用函数。

### 投资组合管理

#### `portfolio.Portfolio`
用于管理资产配置、现金持有和投资组合分析的投资组合类别。

**特征：**
- 体重和现金管理
- 自筹资金约束验证
- 投资组合可视化
- 性能指标计算
- JSON 序列化支持

### 绩效分析

#### `backtest.portfolio_backtester`
用于根据历史数据和基准评估投资组合策略的回测框架。

**支持的方法：**
- 历史数据回测
- KDE（核密度估计）模拟
- 高斯模拟

**指标：**
- 夏普比率
- 索蒂诺比率
- 最大回撤
- 累计回报
- 波动性措施

#### `rebalance.rebalance_portfolio`
具有 CVaR 优化和可配置触发条件的动态投资组合再平衡系统。

**重新平衡触发器：**
- 投资组合漂移阈值
- 性能百分比变化
- 最大回撤限额

**特征：**
- 滚动CVaR优化
- 交易成本建模
- 绩效可视化
- 基线比较

### 数据生成

#### `scenario_generation.ForwardPathSimulator`
使用随机过程生成综合金融数据。

**方法：**
- 几何布朗运动 (log_gbm)
- 前瞻性场景的路径模拟
- 根据历史数据进行校准

### 公用事业

#### `utils`
用于数据处理和投资组合计算的通用实用程序。

**主要功能：**
- `get_input_data()`：多格式数据加载（CSV、Parquet、Excel、JSON）
- `calculate_returns()`：使用对数/线性变换进行返回计算
- `calculate_log_returns()`：日志返回计算
- 性能指标和可视化助手

## 安装

有关安装说明和先决条件，请参阅 [main README](../README.md)。

## 快速入门

### 基本均值-CVaR 优化

#### CVXPY

```python
from src import CvarData, CvarParameters
from src.cvar_optimizer import CVaR
import cvxpy as cp

# Load and prepare return data
returns_dict = {
    'returns': returns_data,  # Historical return scenarios
    'tickers': ['AAPL', 'MSFT', 'GOOGL'],
    'mean': mean_returns,
    'covariance': cov_matrix
}

# Configure optimization parameters
cvar_params = CvarParameters(
    alpha=0.95,                    # CVaR confidence level
    risk_aversion=1.0,             # Risk-return tradeoff
    weight_lower_bound=0.0,        # Min weight per asset
    weight_upper_bound=0.3,        # Max weight per asset
    leverage=1.0                   # No leverage
)

# Create optimizer and solve
optimizer = CVaR(returns_dict, cvar_params)
result, portfolio = optimizer.solve_optimization_problem(
    {"solver": cp.CUOPT}
) #can replace with other CPU solvers
```


#### cuOpt Python API 

```python
# Use cuOpt for GPU acceleration
api_settings = {"api": "cuopt_python"}
optimizer = CVaR(returns_dict, cvar_params, api_settings=api_settings)
result, portfolio = optimizer.solve_optimization_problem({
    "time_limit": 60
})
```

### 回测

```python
from src.backtest import portfolio_backtester

# Initialize backtester
backtester = portfolio_backtester(
    test_portfolio=portfolio,
    returns_dict=returns_dict,
    risk_free_rate=0.02,
    test_method="historical"
)

# Run backtest
metrics = backtester.backtest()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### 动态再平衡

```python
from src.rebalance import rebalance_portfolio

# Configure rebalancing strategy
rebalancer = rebalance_portfolio(
    dataset_directory="data/prices.csv",
    trading_start="2023-01-01",
    trading_end="2024-01-01",
    look_back_window=252,
    look_forward_window=21,
    cvar_params=cvar_params,
    solver_settings={"solver": cp.CLARABEL},
    re_optimize_criteria={
        "type": "drift",
        "threshold": 0.05
    },
    return_type="LOG"
)

# Execute rebalancing strategy
results = rebalancer.rebalance()
```

## 性能考虑因素

- **GPU 加速**：对于具有 100 多个资产或 5000 多个场景的投资组合，cuOpt 可以提供比 CPU 求解器 10-100 倍的加速
- **约束处理**：在 CVXPY 中使用基于参数的约束可以提高热启动性能
- **内存管理**：大型场景集可能需要针对 GPU 内存限制进行分块


## 参考

有关详细的API文档和高级使用示例，请参阅[`notebooks/`](../notebooks/)目录中的jupyter笔记本。

