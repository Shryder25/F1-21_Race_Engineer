"""The race engineer's analysis pipeline, one module per stage.

Each stage is a layer of the one ``AnalysisEngine`` object (analysis.py) and may only call
the stages below it, so the dependency order is explicit and type-checked:

    state      every per-session / per-lap value, typed, with its default
    radio      speak() gating, the busy-driver silence filter, tyre and braking callouts
    models     tyre wear, degradation and fuel (fitted + physics models), cached per lap
    strategy   race plan, undercut / overcut, rival stops, race simulation, Plan B
    track_map  circuit outline, pit lane, live trail, every car on the map
    outputs    web HUD state, history database writes, AI prompt context
    laps       live delta, sectors, lap completion and personal bests, corner diagnosis
    ingest     one handler per game packet: session, motion, status, damage, field, ...

``config`` holds the tuneable settings read from config.toml.
"""
