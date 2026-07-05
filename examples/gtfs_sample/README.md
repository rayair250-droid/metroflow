# Illustrative GTFS sample

This is a **minimal, hand-authored, illustrative** GTFS feed — six fictional
stations on one metro route ("Line M"), both directions — used so MetroFlow's
GTFS ingestion tests and demo run fully offline. **It is not real data** and the
station names, coordinates and times are made up.

Files: `agency.txt`, `stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`,
`calendar.txt` (standard GTFS format).

To build a MetroFlow line from a **real** feed instead, download one and point
MetroFlow at it (see `scripts/fetch_gtfs.py` and the README):

- Île-de-France Mobilités (IDFM) via transport.data.gouv.fr:
  https://transport.data.gouv.fr/datasets/reseau-urbain-et-interurbain-dile-de-france-mobilites
- RATP open data: https://www.ratp.fr/en/ratp-and-open-data

```bash
metroflow gtfs-info examples/gtfs_sample
metroflow simulate --gtfs examples/gtfs_sample --route M1 --direction 0 --seed 42
```
