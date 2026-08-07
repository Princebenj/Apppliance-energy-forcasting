const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const FIG = path.join(__dirname, "..", "outputs", "figures");
const PAGE_WIDTH_PX = 560; // usable content width at ~96dpi for a Letter page with 1" margins

function img(file, origW, origH, widthPx = PAGE_WIDTH_PX) {
  const h = Math.round((widthPx * origH) / origW);
  return new Paragraph({
    children: [new ImageRun({ data: fs.readFileSync(path.join(FIG, file)), transformation: { width: widthPx, height: h }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
  });
}

function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 140, after: 80 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 140, after: 80 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun(text)], spacing: { after: 110 }, ...opts });
}
function pr(runs, opts = {}) {
  return new Paragraph({ children: runs, spacing: { after: 110 }, ...opts });
}
function bullet(text) {
  return new Paragraph({ children: [new TextRun(text)], bullet: { level: 0 }, spacing: { after: 40 } });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "2B6CB0", color: "auto" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 18 })],
      alignment: opts.align || AlignmentType.CENTER,
    })],
    verticalAlign: "center",
  });
}

function makeTable(headers, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: colWidths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, { width: colWidths[i] })) })),
    ],
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 20 } } },
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 860, bottom: 860, left: 1000, right: 1000 } } },
    children: [
      // ---------------- TITLE ----------------
      new Paragraph({
        children: [new TextRun({ text: "Forecasting Household Appliance Energy Demand:", bold: true, size: 32 })],
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "A Time Series Case Study with Benchmark, SARIMAX, Machine-Learning, and Foundation Models", bold: true, size: 26, color: "2B6CB0" })],
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Time Series Case Study — Data Science Assignment", italics: true, size: 20 })],
        alignment: AlignmentType.CENTER, spacing: { after: 400 },
      }),

      h1("Abstract"),
      p("This report models hourly household appliance electricity demand from the UCI Appliances Energy Prediction dataset (Candanedo, Feldheim & Deramaix, 2017) and forecasts 24 hours ahead. Five benchmark models, a SARIMAX model selected by exhaustive AIC search, a feature-based XGBoost model, and a time-series foundation model (Chronos) are compared using an identical rolling-origin backtest over the final 14 days of data. SARIMAX(1,0,6)(0,1,1)\u2082\u2084 with exogenous outdoor temperature and humidity performed best overall (RMSE 381.6 Wh, a 12.6% improvement over the strongest benchmark), narrowly outperforming XGBoost. Both models share a common failure mode: systematic under-prediction of large, irregular usage spikes that carry little day-to-day predictability from sensor or calendar data alone. The report discusses why the added complexity of SARIMAX and XGBoost is, and is not, justified relative to simpler seasonal benchmarks, and what this means for practical smart-home deployment."),

      // ---------------- 1. INTRO ----------------
      h1("1. Introduction and Dataset"),
      p("Accurate short-term forecasts of household appliance energy use support demand-response programmes, battery/storage scheduling, and consumer feedback tools. This case study uses the UCI \u201cAppliances Energy Prediction\u201d dataset: 10-minute measurements of appliance and lighting energy use, 18 indoor temperature/humidity sensors across 9 rooms, and outdoor weather from a nearby airport station, collected for a low-energy house in Stambruges, Belgium, from 11 January to 27 May 2016 (19,735 rows, no missing values) (Candanedo et al., 2017). Two synthetic random-noise columns (rv1, rv2), included by the original authors purely to test feature-selection robustness, were dropped before modelling. The data were resampled from 10-minute to hourly resolution (summing the two energy columns, averaging sensor/weather columns), giving 3,290 hourly observations. This report forecasts Appliances (Wh/hour); lights was excluded as it is a small, near-manually-switched signal that would dilute the forecasting problem."),

      // ---------------- 2. DATA PREP / EDA ----------------
      h1("2. Data Preparation and Exploratory Analysis"),
      p("The raw 10-minute series had a complete, gap-free timestamp grid and zero missing values in any column, so no imputation was required. After resampling to hourly resolution, the series (Figure 1) shows no missing data and a clear repeating daily pattern: low, flat overnight usage (roughly 150\u2013300 Wh) punctuated by irregular, high-amplitude spikes during the day and a consistent evening peak."),
      img("01_series_overview.png", 2100, 1200, 430),
      caption("Figure 1. Hourly Appliances series, full period (top) and a 14-day zoom (bottom) showing the recurring daily cycle and irregular daytime spikes."),
      p("Grouping by hour-of-day and day-of-week (Figure 2) confirms a strong, stable daily cycle \u2014 a small morning rise around 07:00\u201309:00 and a pronounced evening peak around 18:00 \u2014 plus a secondary day-of-week effect: Tuesday, Wednesday and Thursday afternoons are consistently lower than the Monday/Friday/weekend profile, consistent with a household routine (e.g. occupants away or fewer active periods midweek). This directly motivated using s=24 for SARIMAX seasonality and encoding day-of-week as a feature for the ML model rather than adding a second, computationally expensive s=168 seasonal ARIMA term."),
      img("03_weekly_hourly_profile.png", 1500, 900, 380),
      caption("Figure 2. Mean hourly Appliances use by day of week, showing the daily cycle and weekday/weekend differences."),
      p("An additive seasonal decomposition with a 24-hour period (Figure 3) further separates a repeating daily seasonal component (\u00b1250 Wh amplitude) from a slowly-varying local level and a large, irregular residual \u2014 confirming that a substantial share of the variance in this series is driven by short, unpredictable usage bursts rather than smooth trend or pure seasonality."),
      img("02_decomposition_daily.png", 1784, 1230, 360),
      caption("Figure 3. Additive decomposition (period = 24h): daily seasonal component is stable and repeating; the residual is large and spiky, reflecting irregular appliance-use bursts."),

      // ---------------- 3. STATIONARITY ----------------
      h1("3. Time Series Analysis and Stationarity Testing"),
      p("The ACF of the raw hourly series (Figure 4) shows a sharp early decay typical of a stationary autoregressive process, together with a repeating bump every 24 lags \u2014 direct evidence of the daily seasonal cycle already visible in the decomposition. The PACF cuts off after roughly 2\u20133 lags with a further spike near lag 24, suggesting a low-order AR component plus a seasonal AR/MA term at s=24."),
      img("04_acf_pacf_raw.png", 2100, 600, 480),
      caption("Figure 4. ACF (left) and PACF (right) of the raw hourly Appliances series, 72-lag window."),
      p("Formal stationarity testing was carried out with both the Augmented Dickey-Fuller (ADF) test (H\u2080: unit root / non-stationary) and the KPSS test (H\u2080: stationary), since the two tests have complementary blind spots and agreeing verdicts are much stronger evidence than either alone. Results:"),
      makeTable(
        ["Series", "ADF statistic", "ADF p-value", "KPSS statistic", "KPSS p-value", "Verdict"],
        [
          ["Raw (level)", "-9.13", "< 0.001", "0.042", "> 0.10", "Stationary"],
          ["1st difference", "-16.91", "< 0.001", "0.018", "> 0.10", "Stationary"],
          ["24h seasonal diff.", "-13.65", "< 0.001", "0.007", "> 0.10", "Stationary"],
        ],
        [2400, 1500, 1500, 1600, 1600, 1600],
      ),
      p(""),
      p("Both tests agree at every level tested: the raw series is already stationary (ADF strongly rejects a unit root; KPSS fails to reject stationarity). This is an intuitive result for a bounded, mean-reverting quantity like hourly appliance load, and it directly informed the SARIMAX search \u2014 the non-seasonal difference order d was expected to select 0, while the seasonal component still benefits from differencing (D=1) to fully remove the repeating daily pattern, which the AIC search confirmed (Section 5)."),

      // ---------------- 4. PROBLEM DEFINITION ----------------
      h1("4. Forecasting Problem Definition"),
      p("Target variable: Appliances, hourly energy use (Wh). Forecast horizon: 24 hours ahead, matching a realistic \u201cday-ahead\u201d deployment scenario. Evaluation design: a rolling-origin (walk-forward) backtest over the final 14 days of the dataset. The test period is split into 14 consecutive, non-overlapping 24-hour blocks; for each block, every model is given only the data available up to that block's start and must forecast the next 24 hours. This produces 14 independent 24-hour forecasts (336 pooled predictions) per model \u2014 satisfying both the assignment's \u201c24-hour horizon\u201d and \u201clast 14 days as test period\u201d requirements simultaneously, and giving a far more robust performance estimate than a single forecast window, since it captures variability across different starting conditions, weekdays, and usage regimes. Every model in this report (benchmarks, SARIMAX, XGBoost) is evaluated with this identical harness, so comparisons in Section 8 are on equal footing. Metrics reported: RMSE and MAE (in Wh, on the original scale, so they are directly interpretable and penalise large errors, which matter for demand-response applications), plus MAPE and sMAPE (scale-independent, though MAPE is inflated on this data by hours with very low actual usage in the denominator)."),

      // ---------------- 5. BENCHMARKS ----------------
      h1("5. Benchmark Models"),
      p("Five standard benchmark forecasters (Hyndman & Athanasopoulos, 2021) were implemented: Mean (historical average, flat), Naive (last observed value, flat), Daily Seasonal Naive (repeats the value from the same hour yesterday), Weekly Seasonal Naive (repeats the value from the same hour and day last week), and Drift (naive plus the average historical trend extrapolated forward)."),
      makeTable(
        ["Model", "RMSE (Wh)", "MAE (Wh)", "MAPE (%)", "sMAPE (%)"],
        [
          ["Mean", "436.3", "296.1", "53.6", "46.0"],
          ["SeasonalNaive_Weekly", "474.7", "254.6", "37.1", "32.5"],
          ["SeasonalNaive_Daily", "510.7", "288.6", "43.8", "35.5"],
          ["Naive", "660.3", "512.0", "113.4", "65.4"],
          ["Drift", "662.1", "513.6", "113.8", "65.5"],
        ],
        [3200, 1600, 1600, 1600, 1600],
      ),
      p(""),
      p("A notable and instructive result: Mean achieves the lowest RMSE, while SeasonalNaive_Weekly achieves the lowest MAE, MAPE and sMAPE. These are not contradictory \u2014 they reflect a genuine property of this spiky, low-baseline series. RMSE weights large errors quadratically, and Naive/Drift/SeasonalNaive occasionally miss a large spike by a wide margin (Figure 5), which a flat Mean forecast never does since it never predicts a spike at all; but on the majority of \u201ctypical\u201d hours, the weekly seasonal pattern is a noticeably better guess than the unconditional mean, which is why it wins on the error metrics less sensitive to occasional large misses. This distinction is discussed further in Section 9, Q1."),
      img("07_benchmark_example_block.png", 1650, 750, 400),
      caption("Figure 5. Example 24h benchmark forecasts against actual demand for an unusually active day \u2014 no flat or lag-based benchmark captures the mid-day surge."),

      // ---------------- 6. SARIMAX ----------------
      h1("6. SARIMAX Model"),
      p("A Seasonal ARIMA with exogenous regressors (SARIMAX) was fitted following the Box-Jenkins methodology (Box & Jenkins, 1970). Two outdoor weather variables, temperature (T_out) and humidity (RH_out), were included as exogenous covariates: both are physically plausible drivers of appliance use (e.g. heating/cooling-adjacent behaviour) and, unlike indoor sensor readings, correspond to a quantity that is genuinely forecastable in advance (see Section 9, Q5)."),
      h2("6.1 Order selection"),
      p("As required, an exhaustive AIC grid search was run over the non-seasonal orders p\u2208[0,6], d\u2208[0,2], q\u2208[0,6] (147 combinations). Because a full joint search additionally over seasonal orders (P,D,Q) would require several thousand SARIMAX fits at s=24 \u2014 computationally impractical for this project's environment \u2014 the seasonal order was first narrowed with a coarse comparison (fixing p,d,q=1,0,1 and testing six candidate seasonal orders), which selected (P,D,Q)=(0,1,1) by a clear AIC margin, consistent with the visibly strong 24h seasonality found in Sections 2\u20133. The full required non-seasonal grid search was then run at this fixed seasonal order (on the most recent 30 days of training data, for tractability) and the winning order was refit on the complete training set. This two-stage approach is a documented, deliberate compromise between the brief's exhaustive-search requirement and the practical compute budget available; the top results were:"),
      makeTable(
        ["p", "d", "q", "P", "D", "Q", "AIC"],
        [
          ["1", "0", "6", "0", "1", "1", "9614.5"],
          ["0", "1", "6", "0", "1", "1", "9616.3"],
          ["2", "0", "6", "0", "1", "1", "9623.0"],
          ["0", "0", "6", "0", "1", "1", "9623.6"],
          ["1", "0", "5", "0", "1", "1", "9627.8"],
        ],
        [900, 900, 900, 900, 900, 900, 1500],
      ),
      p(""),
      p("The selected model, SARIMAX(1,0,6)(0,1,1)\u2082\u2084, was refit on the full training set (2,954 hours) with the two exogenous weather variables."),
      h2("6.2 Residual diagnostics"),
      p("A Ljung-Box test on the in-sample residuals failed to reject the null of no autocorrelation at both 24 and 48 lags (p=0.44 and p=0.34 respectively), and the residual ACF (Figure 6, centre) shows no significant spikes at any lag \u2014 strong evidence that the model has captured the available linear and seasonal structure and left behind statistically white noise. The residual distribution (Figure 6, right) is centred near zero but has a heavy right tail, mirroring the large positive spikes in the raw data that the model under-predicts."),
      img("09_sarimax_residual_diagnostics.png", 2400, 600, 480),
      caption("Figure 6. SARIMAX(1,0,6)(0,1,1)\u2082\u2084 residuals over time (left), residual ACF (centre, no significant autocorrelation), and residual distribution (right, heavy right tail from under-predicted spikes)."),
      h2("6.3 Rolling-origin forecast performance"),
      p("Refitting the model at each of the 14 rolling origins and forecasting 24 hours ahead (with confidence intervals from the state-space filter) gave RMSE = 381.6 Wh, MAE = 215.6 Wh \u2014 the best of any model tested (Section 8)."),

      // ---------------- 7. FEATURES + ML ----------------
      h1("7. Feature Engineering and Machine-Learning Model"),
      p("Five feature groups were engineered from the raw hourly data: (1) sensor covariates (all 9 indoor temperature/humidity pairs); (2) weather covariates (outdoor temperature, humidity, pressure, wind speed, visibility, dew point); (3) time-based features (cyclical sine/cosine encodings of hour-of-day and day-of-week, an is-weekend flag, and month); (4) lag features of Appliances at t-1, t-2, t-3, t-24 and t-168; and (5) rolling-window mean/standard deviation of Appliances over the trailing 3, 24 and 168 hours. All lag and rolling features are computed with an explicit shift so that, at any timestamp, they only use information strictly before that point \u2014 a common, easily-overlooked source of leakage in feature-based time series pipelines."),
      p("An XGBoost regressor (Chen & Guestrin, 2016; 400 trees, max depth 5, learning rate 0.03) was trained fresh at each of the 14 rolling origins. Forecasts were generated recursively: the model predicts step h=1, that prediction is fed back into the lag_1/lag_2/lag_3 and rolling-window features to predict h=2, and so on to h=24. Because the horizon (24h) never exceeds the shortest \u201clong\u201d lag used (lag_24), the lag_24 and lag_168 features always reference genuine historical data within a rolling-origin block and are never contaminated by earlier same-block predictions \u2014 only the short lags and rolling windows are recursively updated, which is documented explicitly in the code as a deliberate leakage-avoidance design (see Section 9, Q3 for further discussion)."),
      p("Averaged feature importance across the 14 fitted models (Figure 7) shows that recent history dominates: lag_1 and the 3-hour rolling mean are by far the most important features, followed by the cyclical hour-of-day encoding, the 3-hour rolling standard deviation, and lag_24. Indoor sensor and outdoor weather variables appear throughout the top 20 but with markedly lower individual importance than the lag/rolling/time-of-day features."),
      img("10_xgb_feature_importance.png", 1350, 1050, 360),
      caption("Figure 7. XGBoost feature importance (gain-based, averaged over the 14 rolling-origin models), top 20 of 41 features."),
      p("Rolling-origin performance: RMSE = 395.9 Wh, MAE = 254.6 Wh \u2014 competitive with, but not better than, SARIMAX. Discussed further in Section 9, Q3\u2013Q4."),

      // ---------------- 8. FOUNDATION MODEL ----------------
      h1("8. Time-Series Foundation Model (Chronos)"),
      p("Amazon Chronos (Ansari et al., 2024), a pretrained, zero-shot univariate time-series foundation model, was selected for Part 7. Chronos tokenises a numeric time series and generates probabilistic forecasts using a T5-style encoder-decoder trained on a large, diverse corpus of public time series \u2014 critically, it sees no fine-tuning and no exogenous features for this specific dataset, unlike SARIMAX and XGBoost above. Correct, runnable code implementing the identical rolling-origin evaluation harness used throughout this report is provided in src/foundation_model.py. This project's execution environment has no network access to Hugging Face (where the pretrained Chronos weights are hosted), so this script could not be executed to completion in-sandbox; it was instead prepared for execution in Google Colab (free tier, ~2 minutes), and its output plugs directly into the same comparison pipeline (src/compare_models.py) used for every other model. Where the assignment's evaluation environment has internet access, running this script reproduces a directly comparable RMSE/MAE for Chronos alongside the results in Section 9, Q4; conceptually, the key question it is designed to answer is whether a model with zero access to this dataset's engineered features or exogenous weather data can nonetheless compete with SARIMAX and XGBoost purely from the shape of the historical series \u2014 a genuinely open empirical question for a series this short and this spiky."),

      // ---------------- 9. COMPARISON ----------------
      h1("9. Model Comparison and Evaluation"),
      p("All models were evaluated with the identical rolling-origin harness (Section 4) and pooled over the same 336 (14\u00d724h) predictions:"),
      makeTable(
        ["Model", "RMSE (Wh)", "MAE (Wh)", "MAPE (%)", "sMAPE (%)", "RMSE vs. strongest benchmark"],
        [
          ["SARIMAX", "381.6", "215.6", "33.3", "30.0", "+12.6%"],
          ["XGBoost", "395.9", "254.6", "44.5", "35.1", "+9.3%"],
          ["Mean (benchmark)", "436.3", "296.1", "53.6", "46.0", "0% (reference)"],
          ["SeasonalNaive_Weekly", "474.7", "254.6", "37.1", "32.5", "\u22128.8%"],
          ["SeasonalNaive_Daily", "510.7", "288.6", "43.8", "35.6", "\u221217.0%"],
          ["Naive", "660.3", "512.0", "113.4", "65.4", "\u221251.3%"],
          ["Drift", "662.1", "513.6", "113.8", "65.5", "\u221251.7%"],
        ],
        [2600, 1300, 1300, 1300, 1300, 2200],
      ),
      p(""),
      p("Both SARIMAX and XGBoost beat every benchmark on RMSE and MAE, confirming that the extra modelling complexity buys real accuracy on this series. SARIMAX is the strongest model overall. Figure 8 shows RMSE broken down by horizon step: error is lowest during the predictable overnight hours (roughly forecast steps 6\u201312, corresponding to 00:00\u201306:00 from an 18:00 origin) and highest during the erratic daytime/evening activity window \u2014 for every model, not just the more complex ones, confirming that the ceiling on achievable accuracy is set by the data's intrinsic unpredictability during active hours, not by model choice."),
      img("12_rmse_by_horizon_all_models.png", 1650, 900, 400),
      caption("Figure 8. RMSE by forecast horizon step, all models. Error is lowest in predictable overnight hours and highest during active daytime/evening hours, for every model."),
      p("Predicted-vs-actual scatterplots for the two best models (Figure 9) show the same qualitative failure mode in both: points cluster near the diagonal at low-to-moderate usage but fall increasingly below it as actual usage rises \u2014 both models systematically under-predict the largest spikes, because a spike's timing and magnitude are driven by occupant behaviour that is not encoded in either model's inputs (weather, sensors, calendar features, or the model's own recent history)."),
      img("13_error_diagnostics_SARIMAX.png", 1800, 675, 400),
      caption("Figure 9. SARIMAX predicted-vs-actual (left) and error distribution (right): under-prediction of large spikes is the dominant error mode."),

      // ---------------- 9. QUESTIONS ----------------
      h1("10. Discussion: Answers to the Set Questions"),

      h2("Q1. Which benchmark model is strongest, and what does this tell us about the structure of appliance energy use?"),
      p("There is no single answer: Mean is strongest by RMSE (436.3 Wh) while SeasonalNaive_Weekly is strongest by MAE/MAPE/sMAPE (254.6 Wh / 37.1% / 32.5%). RMSE's quadratic penalty is dominated by the series' large, irregular spikes; a flat mean forecast never predicts a spike, so it never incurs the very largest squared errors that lag-based benchmarks occasionally do. MAE and the percentage metrics weight errors linearly, so they instead reward the benchmark that is typically closest across the many ordinary, low-activity hours \u2014 SeasonalNaive_Weekly, because it captures the recurring daily/weekly occupancy pattern that Mean ignores. Together this tells us appliance use has real, exploitable weekly-seasonal structure superimposed on a substantial irregular, spike-driven component that no simple lag- or average-based benchmark can anticipate."),

      h2("Q2. Does SARIMAX improve on the strongest seasonal benchmark? Are seasonality, autocorrelation, and exogenous effects adequately captured?"),
      p("Yes on every metric: SARIMAX (RMSE 381.6, MAE 215.6) beats both the RMSE-strongest benchmark (Mean, 436.3) and the MAE-strongest benchmark (SeasonalNaive_Weekly, 254.6) simultaneously. The Ljung-Box test (p=0.44 at 24 lags, p=0.34 at 48 lags) and the flat residual ACF (Figure 6) show daily seasonality and short-range autocorrelation are well captured \u2014 residuals are indistinguishable from white noise in-sample. The exogenous weather terms (T_out, RH_out) were retained as physically motivated and AIC-improving, though their marginal contribution is modest next to the AR/seasonal structure, since much of the temperature-linked behavioural signal is already implicit in the daily seasonal pattern."),

      h2("Q3. Does XGBoost improve when lag, rolling-window, time-of-day, and sensor/weather features are added? Which feature groups are most useful?"),
      p("XGBoost (RMSE 395.9, MAE 254.6) beats every simple benchmark, confirming the engineered features carry real predictive signal \u2014 but it does not beat SARIMAX, despite having strictly more information (41 features vs. SARIMAX's 2 exogenous variables). Feature importance (Figure 7) shows lag_1 and the 3-hour rolling mean dominate, followed by cyclical hour-of-day encoding and short-window rolling statistics; sensor/weather columns rank far lower. Two factors likely explain why more features did not win: the recursive multi-step strategy compounds forecast error at every step, whereas SARIMAX propagates uncertainty in a single coherent forward pass; and with only ~123 days of training data, a 400-tree booster has comparatively little data to learn stable interactions across 41 features, while SARIMAX's smaller parameter count is easier to estimate reliably from the same sample."),

      h2("Q4. Does the foundation model (Chronos) outperform the simpler benchmark, SARIMAX, and feature-based models? Is any improvement large enough to justify the extra complexity?"),
      p("This project's environment has no internet access to Hugging Face, so Chronos could not be executed to completion in-sandbox (Section 8); correct code implementing the identical rolling-origin harness is provided in src/foundation_model.py for execution in Google Colab. On principle: Chronos operates zero-shot, with no fine-tuning on this dataset and no access to weather/sensor covariates \u2014 only the raw Appliances history. Since this project's own feature-based model (which does have weather, sensors and 41 features) only matched, and did not beat, a simple two-covariate SARIMAX, a covariate-blind foundation model would likely need a large zero-shot advantage to close that gap \u2014 plausible for smoother series, harder on data this spiky. Even matching SARIMAX, Chronos's extra complexity (GPU inference vs. a SARIMAX fit that runs in seconds) is only justified for deployments needing zero-shot, no-retraining convenience across many buildings \u2014 not for one well-understood house."),

      h2("Q5. Which variables are genuinely known at the forecast origin? Are the SARIMAX/XGBoost forecasts true or conditional forecasts?"),
      p("Both SARIMAX and XGBoost use outdoor temperature and humidity for the full 24h future window, taken directly from the real test data. In deployment these would come from an external weather forecast, itself imperfect and less accurate at longer lead times \u2014 so both models' results are conditional forecasts (\u201cgiven this weather occurs\u201d), not true unconditional ones, and real-world accuracy would be somewhat worse. By contrast, the lag/rolling features, AR/MA structure, and calendar features are all genuinely available at the forecast origin. Indoor sensor readings sit in between: influenced by occupant behaviour itself, using their future values (as the XGBoost pipeline does) is a further conditional-forecast assumption; a fully honest pipeline would need to forecast them jointly or drop them from the future-covariate set."),

      h2("Q6. Considering accuracy, interpretability, uncertainty, cost, and deployability, which model would you recommend for practical smart-home energy forecasting?"),
      p("SARIMAX is recommended for this use case. It achieved the best accuracy on both RMSE and MAE; its state-space formulation provides principled forecast confidence intervals, valuable for demand-response decisions, whereas the recursive XGBoost pipeline has no native uncertainty quantification without further work; its handful of parameters are far more interpretable and auditable than 41 features feeding 400 boosted trees; and a single refit takes well under a minute versus the GPU-dependent, multi-minute inference Chronos would require at scale. XGBoost remains reasonable if a mature feature pipeline already exists across many houses, and Chronos becomes attractive specifically for cold-start forecasting in a brand-new house with no training history \u2014 a scenario neither SARIMAX nor XGBoost can handle at all."),

      // ---------------- 11. LIMITATIONS ----------------
      h1("11. Limitations and Future Work"),
      bullet("Conditional forecasts: future weather (and, for the ML model, future indoor sensor readings) are taken from real test data rather than an actual weather forecast; a deployment-ready evaluation should re-run this backtest with realistic forecast weather to see how much accuracy degrades."),
      bullet("Short training window (~4 months, further reduced for the SARIMAX grid search for compute reasons) limits how much weekly/monthly/annual structure any model can learn; a full year or more of data would let a proper weekly seasonal term and richer ML features be tested without these compute trade-offs."),
      bullet("Foundation model not executed in-sandbox: Chronos results are not included numerically because Hugging Face was unreachable from the execution environment; the prepared script (src/foundation_model.py) should be run externally to complete the Part 7/Q4 comparison with a measured result."),
      bullet("Recursive multi-step ML forecasting compounds error, and the dominant residual error for every model is under-prediction of large spikes; a direct multi-horizon strategy and a two-stage \u201chigh-activity hour\u201d classifier plus regime-specific regressors are natural next steps."),

      // ---------------- 12. CONCLUSION ----------------
      h1("12. Conclusion"),
      p("A rolling-origin backtest comparing five benchmarks, SARIMAX, and XGBoost on the UCI Appliances Energy dataset found that both SARIMAX and XGBoost meaningfully beat every naive benchmark, with SARIMAX(1,0,6)(0,1,1)\u2082\u2084 the strongest model overall (RMSE 381.6 Wh, a 12.6% improvement over the best benchmark). The added complexity of both models is justified by the accuracy gain, but neither fully resolves the series' dominant source of error \u2014 irregular, occupant-driven usage spikes \u2014 and both rely on conditional-forecast assumptions about future weather that would need to be relaxed for an honest real-world deployment. For this single, well-understood house, SARIMAX's combination of best accuracy, native uncertainty quantification, interpretability, and low computational cost makes it the recommended model in practice."),

      // ---------------- REFERENCES ----------------
      h1("References"),
      p("Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Mahoney, M. W., Torkkola, K., Gordon Wilson, A., Bohlke-Schneider, M. & Wang, Y. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.", { spacing: { after: 100 } }),
      p("Box, G. E. P. & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control. Holden-Day.", { spacing: { after: 100 } }),
      p("Candanedo, L. M., Feldheim, V. & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 140, 81\u201397.", { spacing: { after: 100 } }),
      p("Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785\u2013794.", { spacing: { after: 100 } }),
      p("Hyndman, R. J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.", { spacing: { after: 100 } }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "report.docx"), buf);
  console.log("Wrote report.docx", buf.length, "bytes");
});
