// Jobs plugin client entry (placeholder).
//
// The real Jobs GUI (job list, create/edit form, run-now button, results
// viewer) lands in a separate change. This stub exists so the plugin's
// client-asset contract is satisfied (the manifest declares client_entry =
// "jobs.js" with assets_dir = "static", and the SPA import()s this module on
// boot for an active plugin). It registers no behaviour yet.
//
// The backend (routes under /api/jobs, the scheduler, the store, and the
// `localm job` CLI) is fully functional without this file.

export function register(_ctx) {
  // No-op until the Jobs GUI ships.
}
