# Data Formats

## Combined Details CSV
Expected key columns:
- `ID`
- `type` (`d` depot, `c` customer, `f` charging point)
- `lng`, `lat`
- `first_receive_tm`, `last_receive_tm`
- `service_time`
- `reward` (for customers)
- `charge_rate_kwh_per_hour` (for `f` and depot rows)

## Distance/Time Matrix CSV
Supported matrix formats:
- Plain numeric square matrix
- ID-labeled matrix with first row/column containing node IDs

For ID-labeled format, row/column ID sets must match.

## Test CSV (`--test-csv`)
Supported schemas:
- Legacy:
  - `customer_id,is_depot,x,y,demand,tw_start,tw_end,service_time,reward,...`
- Combined:
  - `ID,type,lng,lat,first_receive_tm,last_receive_tm,service_time,reward,...`

Charging points can be included directly in test CSV:
- Legacy via `is_charging_station=1`/`is_cp=1`/`node_type=f`
- Combined via `type=f`

When CP rows are present in test CSV, RL uses those rows directly for test-time charging points.
