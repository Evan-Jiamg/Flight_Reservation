# Paper 修訂說明 — 對應 M-1 主網格（540 runs）

對照來源：`EXPERIMENT_LOG.md`（方法規格）、`EXPERIMENT_RESULT.md`（實測結果）、
`Hybrid-Network/analysis/`（圖與 summary）。
基準論文：`Agent_based_Modeling__..._(2).pdf`（2026-07-23 版）。

**所有新寫的內容都以 `revgreen` 標色**（整節用 `\begingroup\color{revgreen} ... \endgroup`，
段落內用 `\textcolor{revgreen}{...}`），編譯後為深綠色，方便與舊文對照。

**先在 `main.tex` 的 preamble 加這一行**（放在 `\usepackage{xcolor}` 之後），否則會編譯失敗：

```latex
\definecolor{revgreen}{RGB}{0,110,55}   % 修訂標示色（深綠，列印友善）
```

不要用 xcolor 內建的 `green`——那是純 #00FF00，螢光刺眼且列印品質差。
要調整色階只需改這一行：更深用 `{0,90,45}`，更亮用 `{0,130,70}`。
定稿時把這行改成 `\definecolor{revgreen}{RGB}{0,0,0}` 即可一次讓全部標示轉回黑色。

**未更動的章節**：Introduction、Sec 2.4（K-NN Rewiring）、Sec 3.3.4（K-NN Dynamic
Rewiring）、Sec 3.4。這四處依指示原封不動。

---

## 1. 本次交付的檔案

| 檔案 | 取代論文的哪一段 |
|---|---|
| `sections__experimental_setup.tex` | Sec 5.1–5.5（Dataset / Configuration / Metrics / Statistics / Implementation） |
| `sections__convergence.tex` | **Sec 4.4 Convergence Analysis —— 目前是空標題** |
| `sections__results.tex` | Sec 5.6 全部（含 Table 1 與 F1–F3） |
| `references_additions.bib` | 6 筆新引用，需附加到 `references.bib` |

`sections__convergence.tex` 是獨立檔，`\input` 到 Sec 4.4 的位置即可，不需要改動
Sec 2.4 / 3.3.4 / 3.4。

**注意 bib key**：我沿用 PDF 參考書目推測的既有 key（`abdin2024phi4`、
`barabasi1999emergence`、`bhawalkar2013coevolutionary`、`blondel2008fast`、
`chitra2020filter`、`gretz2020argq`、`kwon2023vllm`、`newman2004finding`、
`baumgartner2020pushshift`）。若你的 `references.bib` 用的是別的 key，需要替換。
新增的 6 筆是 `tang2010small`、`buttner2016temporal`、`koutra2013deltacon`、
`hubert1985comparing`、`guimera2004modularity`、`fortunato2007resolution`。

---

## 2. 實驗設定的主要變動

| 項目 | 舊論文 | 本次 |
|---|---|---|
| Runs | 180（9 α × 10 seeds × 2 topics） | **540**（+ 3 種初始拓樸） |
| 初始拓樸 | 只有 BA(m=2) | **BA / ER / WS，平均度數對齊（偏差 4.17%）** |
| 模擬長度 | 固定 T = 35 | **動態停止**，`t_conv + 10`，T_max = 120 |
| Type-L 信念 | 5 級離散 {−2…+2}，z = b/2 | **只產生文字**，用 RoBERTa 評分器取連續位置 |
| 評分器 | 直接用 IBM ArgQ 訓練 | **BWS 重新校準**（飽和率 69.7% → 6.5%） |
| Modularity | 裸 Q | **Q_norm（保持度序列的 null model）** |
| 記憶 | 三階段 sliding word window | **模型自行維護**（舊版第 1 步後凍結） |
| 統計 | t-test，df=9 | **Welch + BH-FDR；等價用 TOST（δ=0.2）；跨主題配對** |
| 讀數時點 | t = 35 | **t_conv + post_window**（各 run 自己的收斂態） |

---

## 3. 結論被推翻或需大幅改寫的三處

### 3.1 F1「Pz 幾乎不隨 α 變動」— **反轉**

舊論文：Pz 落在 0.575–0.66，無一致趨勢，據此主張混合比例對意見分散度影響有限。

新資料：**Pz 從 α=0 的 0.304 單調升到 α=1 的 0.597**，Q_norm 從 0.218 升到 0.423，
兩者 CI 都很窄。舊的「平坦」是 5 級量化的產物 —— 只有五個可用位置時，族群變異數
被格距撐住了。

**影響**：F1 必須拆成兩句。「同溫層由 K-NN 造成、不需預先存在的 hub 結構」這部分
成立（由三拓樸的結果支持）；但「agent 類型與結構強度無關」不成立。RQ2 的回答要改寫。

### 3.2 F3「norm contention cost 峰值在中高 α、跨主題一致」— **位置與範圍都錯**

舊論文：峰值在 α=0.5–0.75，abortion 達 3.364，宣稱兩主題一致。

新資料：**峰值在 α=0.125，且只有 abortion 有**（p=0.008、d=0.71）；gun_control
完全沒有峰值，最大值就在 α=0。α ≤ 0.375 是統計上的平台，之後才單調顯著下降。

**根因**：舊結果在 t=35 讀數，但 α ≤ 0.5 的 run 平均要 35–52 步才收斂 —— 舊的
「中高 α 峰值」比較的是「已收斂的高 α」對「還在暫態的低 α」。

### 3.3 「Q ≈ 0.76 代表強同溫層」— **高估兩倍以上**

裸 Q 中約 44% 是保持度序列的隨機圖也會產生的。應改述為 Q_norm ≈ 0.42–0.43。
結構仍顯著非隨機（z_Q ≈ 24–26），但強度的表述必須改。

---

## 4. 我沒有改、但現在與新資料衝突的章節

以下都在你指定的「可動」範圍外，或屬於需要你決定語氣的部分，我列出來但沒有動筆：

### Abstract
- 「Purely LLM populations maintain **comparable or slightly higher polarization** than
  numerical populations」→ **與新資料相反**。LLM 族群的極化明顯**較低**（0.304 vs 0.597）。
- 「scored on a continuous [−1,+1] scale via a fine-tuned RoBERTa regressor」→ 需說明
  這是**重新校準後**的評分器，舊的並非真正連續。
- 「Numerical-heavy populations are more efficient, but produce stronger echo chamber
  structures」→ 這句**仍然成立**，而且新資料支持得更乾淨（兩條單調曲線）。

### Sec 6.1 Interpretation of Key Findings
- 引用的 PoA 數字（2.807 / 2.553 / 3.364）全部作廢，需換成 5.68 / 5.44 / 6.32。
- 「Modularity (Q = 0.646 / 0.542) and polarization (Pz ≈ 0.65) are already substantial
  at α = 0」→ 數字與詮釋都要改（Q_norm 0.241 / 0.194，Pz 0.261 / 0.347）。
- **可以加強的一段**：舊文只能說「LLM 更容易被說服」是推論，新資料用成本拆解
  直接量到了（從眾佔比 43.6% vs 16.1%，相差 13.2 倍）。這是本次最值得寫進 Discussion 的一項。

### Sec 6.2 Research Questions Revisited
- RQ1 的「non-monotone、abortion 在 α=0.25 有 local minimum、0.5–0.75 有峰值」整段作廢。
- RQ2 的「Pz varies only modestly across all α，no consistent monotone trend」**直接相反**。
- RQ3 的「α = 1 optimal for efficiency」**仍成立**。

### Sec 6.3 Limitations and Future Work
三條限制**有兩條已經被本次實驗解掉**，應該改寫而不是保留：
- 第二條「只跑了 scale-free，未來應比較 scale-free / small-world / random」→ **已完成**，
  應改成結果陳述（三拓樸無顯著差異，α=1 經 TOST 證實等價）。
- 第三條「Type-L 用固定五點離散尺度…未來應研究連續信念表示」→ **已完成**。
- 第一條（N=50、未擴展到更大族群）**仍然成立**，保留。

新的限制應該補上：
- CI 只反映「拓樸與分派」的變異，**不含「抽到哪 50 個 Reddit 使用者」的抽樣變異**。
- 低 α 的三拓樸等價**未獲 TOST 證實**（樣本量不足，δ=0.2 在低 α 端只有 3.6% 相對界限）。
- Louvain 的**解析極限**：K=5、N=50 下界約 √(2m) ≈ 22，而實測最小社群只有 5–7，
  null model 無法處理此問題。
- P_MAX = 20，週期大於 20 者被歸為非週期。
- 兩主題共用人設與頑固度是**控制變因用的簡化**，非實證推導。
- leader 由 index `[n//5, n*3//5]` 指定，且現行 prompt 的語意是「頑固的廣播者」
  而非「有影響力的節點」—— 這點論文中目前沒有交代。

### Sec 7 Conclusion
所有數字需更新；「polarization remains relatively stable with only a modest decline」
需反轉為單調上升。

---

## 5. 圖：全部需要重做（這是目前最大的缺口）

**論文現有的 Figure 4、5、6、7 都不能用**，它們是 6 月 21 日、T=35、180-run 的產物：

| 現有圖 | 狀態 |
|---|---|
| Fig 4 `convergence_summary_bar.png` | 舊資料。而且它畫的是「±5% settling step」，與本次的 `t_conv` 判準定義不同，不可混用 |
| Fig 5 `compare_topics.png` / `compare_lines.png` | 舊資料 |
| Fig 6/7 `{topic}_timeseries.png` | 舊資料，且橫軸固定 35 步 |
| Fig 2/3（Reddit 分佈） | **需用 BWS 重新校準後的評分重畫**，舊圖是飽和評分器的輸出 |

`analysis/figures/` 下 07-28 產生的 27 張 `hybrid_{network}_K5_alpha{a}_agents50.png`
**也不適合直接當論文圖**：它們是單 run 診斷圖，畫的是裸 Q、沒有 CI 帶、沒有跨 seed 聚合。

`EXPERIMENT_LOG.md` §3.9 已經規格化了應該畫的六張圖（F1–F6），但**尚未產生**：

| 圖 | 內容 |
|---|---|
| F1 | `C_out(t)` 曲線，mean ± 95% CI，欄=網路、列=主題，標 t_conv 中位數（**主圖**） |
| F2 | `dS(t)`、`dL(t)`、`1 − DeltaCon(t)`，log-y（多指標交叉佐證） |
| F3 | `t_conv` 累積分佈，分 α 上色（**核心證據**） |
| F4 | `Pz(t)`、`Q_norm(t)`、`PoA(t)` 含 CI 帶，標 t_conv |
| F5 | attractor 型態堆疊長條圖，x = α |
| F6 | 相圖 (Pz(t), Q(t))，以 t 上色 |

另外建議補一張本次特有的：**PoA 成分拆解的堆疊面積圖**（不合 vs 從眾，x = α），
這是 F2 finding 最直接的視覺化。

資料都在 `/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4`，
不需要重跑模擬，只需要寫繪圖腳本。

---

## 6. 兩個需要你決定的方法問題

1. **低 α 的等價主張**：要不要對 α < 1 主張三拓樸等價？若要，必須改用相對界限
   （例如 5%，δ ≈ 0.28）或補 seeds 11–30。`EXPERIMENT_LOG.md` §2 明訂這是事前決策，
   不宜看過資料再挑。目前我在 results 裡採**保守版**（只在 α=1 主張等價）。

2. **主題要不要合併呈現**：兩主題不是獨立樣本（corr = +0.881，共用人設與 rho）。
   我在 Table 1 分開列、在 α 曲線用合併值（n=60）。若審稿人質疑合併的合法性，
   替代方案是全部分開呈現，但 CI 會寬約 √2 倍。
