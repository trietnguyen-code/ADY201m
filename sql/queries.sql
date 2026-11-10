-- Step 3: sql/queries.sql
CREATE OR REPLACE TABLE d AS SELECT * FROM 'data/processed/pmdata_daily.parquet';
CREATE OR REPLACE TABLE h AS SELECT * FROM 'data/processed/pmdata_hourly.parquet';

-- Q1: reports and typical readiness per person (how different people are from each other)
SELECT pid, COUNT(readiness) AS reports, AVG(readiness) AS readiness, STDDEV(readiness) AS sd, AVG(steps) AS steps,
       AVG(load) AS load FROM d GROUP BY 1 ORDER BY 1;

-- Q2: next-morning readiness after hard, light and rest days (LEAD gives the report of the following day)
WITH n AS (SELECT *, LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_ready,
                  LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date FROM d)
SELECT CASE WHEN load = 0 THEN 'rest' WHEN load < 400 THEN 'light' ELSE 'hard' END AS day_type, COUNT(*) AS n,
       AVG(next_ready) AS next_readiness, AVG(sleep_min) AS sleep_min
FROM n WHERE next_ready IS NOT NULL AND next_date = date + INTERVAL 1 DAY GROUP BY 1 HAVING COUNT(*) > 20 ORDER BY 3;

-- Q3: hourly step profile, weekday against weekend (hourly table, the large one)
SELECT HOUR(ts) AS hour, CASE WHEN DAYOFWEEK(ts) IN (0, 6) THEN 'weekend' ELSE 'weekday' END AS part, AVG(steps) AS steps
FROM h GROUP BY 1, 2 ORDER BY 1, 2;

-- Q4: model table; target is tomorrow's readiness, every feature is known by tonight (past-only windows)
CREATE OR REPLACE TABLE feat AS
WITH b AS (
  SELECT *, LEAD(readiness) OVER o AS y, LEAD(date) OVER o AS next_date, ROW_NUMBER() OVER o AS day_no,
         AVG(load) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS load_7d,
         AVG(load) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS load_28d,
         AVG(sleep_min) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS sleep_7d,
         AVG(readiness) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ready_base,
         AVG(rhr) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rhr_base,
         AVG(steps) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS steps_base
  FROM d WINDOW o AS (PARTITION BY pid ORDER BY date))
SELECT *, load_7d / NULLIF(load_28d, 0) AS load_ratio, readiness - ready_base AS ready_dev, rhr - rhr_base AS rhr_dev,
       steps - steps_base AS steps_dev, DAYOFWEEK(date) AS dow
FROM b WHERE y IS NOT NULL AND readiness IS NOT NULL AND next_date = date + INTERVAL 1 DAY AND day_no > 7;
SELECT COUNT(*) AS rows, COUNT(DISTINCT pid) AS people FROM feat;

-- Q5: multi-modal compliance and missingness rates across participants (Fitbit telemetry vs. PMSys self-reports)
SELECT pid,
       COUNT(*) AS recorded_days,
       COUNT(readiness) AS readiness_reports,
       ROUND(COUNT(readiness) * 100.0 / COUNT(*), 1) AS survey_compliance_pct,
       COUNT(steps) AS step_days,
       COUNT(sleep_min) AS sleep_days,
       COUNT(rhr) AS rhr_days,
       ROUND(AVG(readiness), 2) AS mean_readiness,
       ROUND(STDDEV(readiness), 2) AS sd_readiness
FROM d
GROUP BY pid
ORDER BY pid;

-- Q6: acute-to-chronic workload ratio (ACWR) zones and autonomic resting heart rate deviation
WITH rolled AS (
  SELECT pid, date, load, rhr, readiness,
         AVG(load) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS acute_load,
         AVG(load) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS chronic_load,
         AVG(rhr) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rhr_baseline,
         LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_ready,
         LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date
  FROM d
),
binned AS (
  SELECT pid, date,
         acute_load / NULLIF(chronic_load, 0) AS acwr,
         rhr - rhr_baseline AS rhr_dev,
         next_ready
  FROM rolled
  WHERE next_date = date + INTERVAL 1 DAY AND next_ready IS NOT NULL AND chronic_load > 0
)
SELECT CASE WHEN acwr < 0.8 THEN '1. Under-load (< 0.8)'
            WHEN acwr BETWEEN 0.8 AND 1.3 THEN '2. Optimal zone (0.8 - 1.3)'
            WHEN acwr BETWEEN 1.3 AND 1.5 THEN '3. Moderate spike (1.3 - 1.5)'
            ELSE '4. High spike (> 1.5)' END AS acwr_zone,
       COUNT(*) AS transition_count,
       ROUND(AVG(acwr), 2) AS mean_acwr,
       ROUND(AVG(rhr_dev), 2) AS mean_rhr_dev_bpm,
       ROUND(AVG(next_ready), 2) AS mean_next_readiness
FROM binned
GROUP BY 1
ORDER BY 1;

-- Q7: sleep duration tiers and subjective sleep quality on prospective athletic readiness
WITH sleep_cohort AS (
  SELECT pid, date,
         sleep_min / 60.0 AS sleep_hours,
         sleep_quality,
         readiness,
         LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_readiness,
         LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date
  FROM d
  WHERE sleep_min IS NOT NULL AND readiness IS NOT NULL
)
SELECT CASE WHEN sleep_hours < 6.0 THEN 'Short sleep (< 6h)'
            WHEN sleep_hours BETWEEN 6.0 AND 8.0 THEN 'Normal sleep (6h - 8h)'
            ELSE 'Extended sleep (> 8h)' END AS sleep_tier,
       CASE WHEN sleep_quality <= 2 THEN 'Low quality (1-2)'
            WHEN sleep_quality = 3 THEN 'Moderate quality (3)'
            ELSE 'High quality (4-5)' END AS quality_tier,
       COUNT(*) AS count_sessions,
       ROUND(AVG(sleep_hours), 2) AS avg_sleep_hrs,
       ROUND(AVG(readiness), 2) AS same_day_readiness,
       ROUND(AVG(next_readiness), 2) AS next_day_readiness
FROM sleep_cohort
WHERE next_date = date + INTERVAL 1 DAY AND next_readiness IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2;

-- Q8: day-of-week behavioral and recovery rhythms (training load, sleep, steps, readiness cyclicity)
SELECT DAYNAME(date) AS day_name,
       DAYOFWEEK(date) AS day_index,
       COUNT(*) AS recorded_days,
       ROUND(AVG(readiness), 2) AS mean_readiness,
       ROUND(AVG(steps), 0) AS mean_steps,
       ROUND(AVG(very_active), 1) AS mean_active_min,
       ROUND(AVG(load), 1) AS mean_training_load,
       ROUND(AVG(sleep_min) / 60.0, 2) AS mean_sleep_hours
FROM d
WHERE readiness IS NOT NULL
GROUP BY 1, 2
ORDER BY day_index;

-- Q9: consecutive transition continuity and calendar gap distribution (verification of leak-free pairing)
WITH daily_gaps AS (
  SELECT pid, date,
         LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date,
         LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_readiness,
         readiness AS cur_readiness
  FROM d
),
gap_calc AS (
  SELECT pid, date,
         DATE_DIFF('day', date, next_date) AS calendar_gap_days,
         CASE WHEN DATE_DIFF('day', date, next_date) = 1 THEN '1. Consecutive (t -> t+1)'
              WHEN DATE_DIFF('day', date, next_date) = 2 THEN '2. 1-day missed (gap=2)'
              WHEN DATE_DIFF('day', date, next_date) BETWEEN 3 AND 7 THEN '3. Weekly gap (3-7 days)'
              ELSE '4. Long absence (> 7 days)' END AS continuity_type,
         CASE WHEN cur_readiness IS NOT NULL AND next_readiness IS NOT NULL THEN 1 ELSE 0 END AS both_surveys_valid
  FROM daily_gaps
  WHERE next_date IS NOT NULL
)
SELECT continuity_type,
       COUNT(*) AS total_transitions,
       SUM(both_surveys_valid) AS valid_paired_surveys,
       ROUND(SUM(both_surveys_valid) * 100.0 / COUNT(*), 1) AS valid_pairing_rate_pct
FROM gap_calc
GROUP BY 1
ORDER BY 1;

-- Q5: multi-modal compliance and missingness rates across participants (Fitbit telemetry vs. PMSys self-reports)
SELECT pid,
       COUNT(*) AS recorded_days,
       COUNT(readiness) AS readiness_reports,
       ROUND(COUNT(readiness) * 100.0 / COUNT(*), 1) AS survey_compliance_pct,
       COUNT(steps) AS step_days,
       COUNT(sleep_min) AS sleep_days,
       COUNT(rhr) AS rhr_days,
       ROUND(AVG(readiness), 2) AS mean_readiness,
       ROUND(STDDEV(readiness), 2) AS sd_readiness
FROM d
GROUP BY pid
ORDER BY pid;

-- Q6: acute-to-chronic workload ratio (ACWR) zones and autonomic resting heart rate deviation
WITH rolled AS (
  SELECT pid, date, load, rhr, readiness,
         AVG(load) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS acute_load,
         AVG(load) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS chronic_load,
         AVG(rhr) OVER (PARTITION BY pid ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rhr_baseline,
         LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_ready,
         LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date
  FROM d
),
binned AS (
  SELECT pid, date,
         acute_load / NULLIF(chronic_load, 0) AS acwr,
         rhr - rhr_baseline AS rhr_dev,
         next_ready
  FROM rolled
  WHERE next_date = date + INTERVAL 1 DAY AND next_ready IS NOT NULL AND chronic_load > 0
)
SELECT CASE WHEN acwr < 0.8 THEN '1. Under-load (< 0.8)'
            WHEN acwr BETWEEN 0.8 AND 1.3 THEN '2. Optimal zone (0.8 - 1.3)'
            WHEN acwr BETWEEN 1.3 AND 1.5 THEN '3. Moderate spike (1.3 - 1.5)'
            ELSE '4. High spike (> 1.5)' END AS acwr_zone,
       COUNT(*) AS transition_count,
       ROUND(AVG(acwr), 2) AS mean_acwr,
       ROUND(AVG(rhr_dev), 2) AS mean_rhr_dev_bpm,
       ROUND(AVG(next_ready), 2) AS mean_next_readiness
FROM binned
GROUP BY 1
ORDER BY 1;

-- Q7: sleep duration tiers and subjective sleep quality on prospective athletic readiness
WITH sleep_cohort AS (
  SELECT pid, date,
         sleep_min / 60.0 AS sleep_hours,
         sleep_quality,
         readiness,
         LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_readiness,
         LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date
  FROM d
  WHERE sleep_min IS NOT NULL AND readiness IS NOT NULL
)
SELECT CASE WHEN sleep_hours < 6.0 THEN 'Short sleep (< 6h)'
            WHEN sleep_hours BETWEEN 6.0 AND 8.0 THEN 'Normal sleep (6h - 8h)'
            ELSE 'Extended sleep (> 8h)' END AS sleep_tier,
       CASE WHEN sleep_quality <= 2 THEN 'Low quality (1-2)'
            WHEN sleep_quality = 3 THEN 'Moderate quality (3)'
            ELSE 'High quality (4-5)' END AS quality_tier,
       COUNT(*) AS count_sessions,
       ROUND(AVG(sleep_hours), 2) AS avg_sleep_hrs,
       ROUND(AVG(readiness), 2) AS same_day_readiness,
       ROUND(AVG(next_readiness), 2) AS next_day_readiness
FROM sleep_cohort
WHERE next_date = date + INTERVAL 1 DAY AND next_readiness IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2;

-- Q8: day-of-week behavioral and recovery rhythms (training load, sleep, steps, readiness cyclicity)
SELECT DAYNAME(date) AS day_name,
       DAYOFWEEK(date) AS day_index,
       COUNT(*) AS recorded_days,
       ROUND(AVG(readiness), 2) AS mean_readiness,
       ROUND(AVG(steps), 0) AS mean_steps,
       ROUND(AVG(very_active), 1) AS mean_active_min,
       ROUND(AVG(load), 1) AS mean_training_load,
       ROUND(AVG(sleep_min) / 60.0, 2) AS mean_sleep_hours
FROM d
WHERE readiness IS NOT NULL
GROUP BY 1, 2
ORDER BY day_index;

-- Q9: consecutive transition continuity and calendar gap distribution (verification of leak-free pairing)
WITH daily_gaps AS (
  SELECT pid, date,
         LEAD(date) OVER (PARTITION BY pid ORDER BY date) AS next_date,
         LEAD(readiness) OVER (PARTITION BY pid ORDER BY date) AS next_readiness,
         readiness AS cur_readiness
  FROM d
),
gap_calc AS (
  SELECT pid, date,
         DATE_DIFF('day', date, next_date) AS calendar_gap_days,
         CASE WHEN DATE_DIFF('day', date, next_date) = 1 THEN '1. Consecutive (t -> t+1)'
              WHEN DATE_DIFF('day', date, next_date) = 2 THEN '2. 1-day missed (gap=2)'
              WHEN DATE_DIFF('day', date, next_date) BETWEEN 3 AND 7 THEN '3. Weekly gap (3-7 days)'
              ELSE '4. Long absence (> 7 days)' END AS continuity_type,
         CASE WHEN cur_readiness IS NOT NULL AND next_readiness IS NOT NULL THEN 1 ELSE 0 END AS both_surveys_valid
  FROM daily_gaps
  WHERE next_date IS NOT NULL
)
SELECT continuity_type,
       COUNT(*) AS total_transitions,
       SUM(both_surveys_valid) AS valid_paired_surveys,
       ROUND(SUM(both_surveys_valid) * 100.0 / COUNT(*), 1) AS valid_pairing_rate_pct
FROM gap_calc
GROUP BY 1
ORDER BY 1;

