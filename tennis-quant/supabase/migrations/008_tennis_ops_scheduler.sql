create extension if not exists pg_net with schema extensions;
create extension if not exists pg_cron;

do $$
begin
  if exists (
    select 1 from cron.job where jobname = 'tennis-clv-hourly'
  ) then
    perform cron.unschedule('tennis-clv-hourly');
  end if;

  if exists (
    select 1 from cron.job where jobname = 'tennis-health-six-hourly'
  ) then
    perform cron.unschedule('tennis-health-six-hourly');
  end if;
end
$$;

select cron.schedule(
  'tennis-clv-hourly',
  '20 * * * *',
  $$
    select net.http_get(
      url := 'https://moteur-lemon.vercel.app/api/cron/tennis-clv',
      params := '{}'::jsonb,
      headers := jsonb_build_object(
        'Authorization',
        'Bearer ' || (
          select decrypted_secret
          from vault.decrypted_secrets
          where name = 'tennis_ops_scheduler_token'
          limit 1
        )
      ),
      timeout_milliseconds := 280000
    );
  $$
);

select cron.schedule(
  'tennis-health-six-hourly',
  '50 */6 * * *',
  $$
    select net.http_get(
      url := 'https://moteur-lemon.vercel.app/api/cron/tennis-health',
      params := '{}'::jsonb,
      headers := jsonb_build_object(
        'Authorization',
        'Bearer ' || (
          select decrypted_secret
          from vault.decrypted_secrets
          where name = 'tennis_ops_scheduler_token'
          limit 1
        )
      ),
      timeout_milliseconds := 280000
    );
  $$
);
