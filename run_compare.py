import argparse
import os
import json
from eval import evaluate

def run_and_save(args, checkpoint, tag):
    # build args for evaluate
    ev_args = argparse.Namespace(
        data_root=args.data_root,
        checkpoint=checkpoint,
        vocab_file=args.vocab_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        dec_dim=args.dec_dim,
        attn_dim=args.attn_dim,
        pin_memory=args.pin_memory,
        output_dir=os.path.join(args.output_dir, tag),
        force_cpu=args.force_cpu,
        force_bleu=args.force_bleu,
        beam_size=args.beam_size,
        no_meteor=args.no_meteor,
        no_rouge=args.no_rouge
    )
    metrics, hyps, refs = evaluate(ev_args)
    return metrics, hyps, refs

def make_table(example_ids, refs, hyps_xe, hyps_scst):
    rows = []
    for i in example_ids:
        gt = refs.get(i, [""])[0] if isinstance(refs.get(i, []), list) and len(refs.get(i, []))>0 else ""
        xe = hyps_xe.get(i, "")
        scst = hyps_scst.get(i, "")
        rows.append((i, gt, xe, scst))
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--checkpoint_xe', required=True, help='checkpoint for XE-trained model')
    parser.add_argument('--checkpoint_scst', required=True, help='checkpoint for SCST-trained model')
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--dec_dim', type=int, default=512)
    parser.add_argument('--attn_dim', type=int, default=512)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--output_dir', type=str, default='compare_outputs')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--force_bleu', action='store_true')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--no_meteor', action='store_true')
    parser.add_argument('--no_rouge', action='store_true')
    args = parser.parse_args()

    m_xe, hyps_xe, refs_xe = run_and_save(args, args.checkpoint_xe, 'XE')
    m_scst, hyps_scst, refs_scst = run_and_save(args, args.checkpoint_scst, 'SCST')

    # refs_xe and refs_scst should be same; pick keys
    sample_keys = list(refs_xe.keys())[:10]

    table = make_table(sample_keys, refs_xe, hyps_xe, hyps_scst)

    # print metrics summary
    print("METRICS SUMMARY")
    print("XE metrics:")
    for k,v in m_xe.items():
        print(f"  {k}: {v}")
    print("SCST metrics:")
    for k,v in m_scst.items():
        print(f"  {k}: {v}")

    print("\nSAMPLE CAPTIONS (image_name | GT | XE | SCST):")
    for r in table:
        print("---")
        print("Image:", r[0])
        print("GT:", r[1])
        print("XE:", r[2])
        print("SCST:", r[3])

    # save CSV-like json for easy inspection
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "compare_summary.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'metrics_xe': m_xe, 'metrics_scst': m_scst, 'samples': [{ 'img': r[0], 'gt': r[1], 'xe': r[2], 'scst': r[3]} for r in table]}, f, ensure_ascii=False, indent=2)
    print("Saved comparison to", out_path)