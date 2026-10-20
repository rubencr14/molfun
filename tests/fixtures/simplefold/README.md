# SimpleFold test fixtures

- `tiny_record/` — PDB 1CRN (crambin, 46 residues, X-ray) processed with the
  pinned SimpleFold fork (`62f1a22`):

  ```bash
  redis-server --dbfilename ccd.rdb --dir $MOLFUN_WORK_DIR/boltz --port 7781 --save "" --daemonize yes
  cd <fork>/src/simplefold
  python process_mmcif.py --data_dir <dir with 1crn.cif> --out_dir tiny_record/targets --num-processes 1 --redis-port 7781
  python process_structure.py --target_dir tiny_record/targets --token_dir tiny_record/tokens
  redis-cli -p 7781 shutdown nosave
  ```

- `tiny_arch.yaml` — a FoldingDiT small enough for CPU unit tests
  (hidden 64, trunk depth 2), conditioned on ESM2-8M.
