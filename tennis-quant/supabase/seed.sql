insert into public.tennis_model_runs
  (tour, model_name, model_version, trained_through, validation_method, brier_score, log_loss, sample_size, artifact_uri)
select *
from (
  values
    ('ATP','full_logit','v1-oos-2022-2025','2026-05-25'::date,'walk-forward',0.21545884548052083::numeric,0.6189421338712963::numeric,10514,'github://Matttgic/Moteur/tennis-quant/ml/benchmark_reference.json'),
    ('WTA','full_logit','v1-oos-2022-2025','2026-05-25'::date,'walk-forward',0.21260395436693225::numeric,0.6134117991289917::numeric,4485,'github://Matttgic/Moteur/tennis-quant/ml/benchmark_reference.json')
) as seed(tour, model_name, model_version, trained_through, validation_method, brier_score, log_loss, sample_size, artifact_uri)
where not exists (
  select 1
  from public.tennis_model_runs r
  where r.tour = seed.tour
    and r.model_version = seed.model_version
);
