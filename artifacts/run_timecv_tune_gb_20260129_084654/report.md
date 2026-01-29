# GradientBoosting time-CV tuning

## Best configuration (by mean F2)
{
  "n_estimators": 100,
  "learning_rate": 0.1,
  "max_depth": 2,
  "min_samples_leaf": 5,
  "subsample": 0.85
}

## Operating points (evaluated on test segments)
- f2_opt: threshold=0.025523, precision=0.4762, recall=0.9434, f2=0.7886, alert_rate=0.1400, alerts_per_1000=140.0
- high_recall: threshold=0.035156, precision=0.5102, recall=0.9434, f2=0.8065, alert_rate=0.1307, alerts_per_1000=130.7
- alert_budget_5p: threshold=0.850872, precision=1.0000, recall=0.3962, f2=0.4506, alert_rate=0.0280, alerts_per_1000=28.0

## Top features (mean importance)
- congestion: 0.2970 ± 0.0273
- packet_loss: 0.2759 ± 0.0226
- latency: 0.1213 ± 0.0079
- jitter: 0.1199 ± 0.0042
- throughput: 0.1003 ± 0.0123
- congestion__lag3: 0.0166 ± 0.0060
- throughput__lag2: 0.0151 ± 0.0055
- congestion__lag1: 0.0117 ± 0.0079
- throughput__lag3: 0.0069 ± 0.0069
- congestion__roll6_std: 0.0055 ± 0.0013

## Example true positives
- timestamp=2024-05-11T22:17:53+00:00, score=0.9988, top_features: congestion: value=110.190, roll3_mean=24.453, delta1=110.100; packet_loss: value=52.500, roll3_mean=8.333, delta1=27.500; throughput: value=0.110, roll3_mean=2.027, delta1=-1.480
- timestamp=2024-05-11T22:17:53+00:00, score=0.9988, top_features: congestion: value=110.190, roll3_mean=24.453, delta1=110.100; packet_loss: value=52.500, roll3_mean=8.333, delta1=27.500; throughput: value=0.110, roll3_mean=2.027, delta1=-1.480
- timestamp=2024-05-11T22:17:53+00:00, score=0.9982, top_features: congestion: value=110.190, roll3_mean=24.453, delta1=110.100; packet_loss: value=52.500, roll3_mean=8.333, delta1=27.500; throughput: value=0.110, roll3_mean=2.027, delta1=-1.480

## Example false positives
- timestamp=2024-05-11T22:04:52+00:00, score=0.1814, top_features: packet_loss: value=27.500, roll3_mean=7.500, delta1=27.500; congestion: value=63.210, roll3_mean=62.363, delta1=17.040; latency: value=9.860, roll3_mean=10.343, delta1=-0.400
- timestamp=2024-05-11T23:08:55+00:00, score=0.1578, top_features: congestion: value=85.480, roll3_mean=65.440, delta1=24.250; packet_loss: value=22.500, roll3_mean=6.667, delta1=2.500; latency: value=11.950, roll3_mean=9.473, delta1=2.090
- timestamp=2024-05-11T22:27:49+00:00, score=0.1505, top_features: congestion: value=68.160, roll3_mean=0.050, delta1=68.100; packet_loss: value=27.500, roll3_mean=10.833, delta1=17.500; latency: value=10.150, roll3_mean=10.763, delta1=-0.650