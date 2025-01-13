import torch
from math import sqrt

torch.random.manual_seed(42)

from flashinfer import BatchDecodeMlaWithPagedKVCacheWrapper

FLASHINFER_WORKSPACE_BUFFER_SIZE = 25 * 1024 * 1024

workspace = torch.empty(FLASHINFER_WORKSPACE_BUFFER_SIZE,
                        dtype=torch.uint8,
                        device="cuda")


# [total_blocks, 2, block_size, num_heads, head_dim] we store the rope cache in the "value" section
kv_cache = torch.zeros([2, 2, 16, 1, 512], dtype=torch.bfloat16, device="cuda")
query = torch.zeros([2, 16, 576], dtype=torch.bfloat16, device="cuda")

# here we load the real activations from a sample query using DeepseekV2-Lite-Chat
import safetensors.torch
import sys
if len(sys.argv) > 1:
    path = sys.argv[1]
    print(f"Using weights from {path}")
    state_dict = safetensors.torch.load_file(path)
    q_pe = state_dict["q_pe"]
    q_nope = state_dict["q_nope"]
    k_pe_cache = state_dict["k_pe_cache"]
    compressed_kv_normed_cache = state_dict["compressed_kv_normed_cache"]
    # print(f"{q_pe.shape=}, {q_noope.shape=}, {k_pe_cache.shape=}, {compressed_kv_normed_cache.shape=}")
    # q_pe.shape=torch.Size([2, 16, 64]), q_nope.shape=torch.Size([2, 16, 512]), k_pe_cache.shape=torch.Size([2, 9, 64]), compressed_kv_normed_cache.shape=torch.Size([2, 9, 512])
    query[:, :, :512] = q_nope
    query[:, :, 512:] = q_pe
    kv_cache[:, 0, :9, 0, :] = compressed_kv_normed_cache
    kv_cache[:, 1, :9, 0, :64] = k_pe_cache
    kv_cache[:, 1, :9, 0, 64:] = 0
else:
    print("Randomly initializing")
    kv_cache = torch.randn([2, 2, 16, 1, 512], dtype=torch.bfloat16, device="cuda")
    query = torch.randn([2, 16, 576], dtype=torch.bfloat16, device="cuda")


wrapper = BatchDecodeMlaWithPagedKVCacheWrapper(workspace)

wrapper.plan(
    indptr=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
    indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
    last_page_len=torch.tensor([7, 9], dtype=torch.int32, device="cuda"),
    num_qo_heads=16,
    head_dim_compressed_kv=512,
    page_size=16,
    sm_scale=1 / sqrt(512 + 64),
    data_type=query.dtype,
)

head_size = 512
q_nope = query[:, :, :head_size].contiguous()
assert q_nope.shape == (2, 16, 512)
q_pe = query[:, :, head_size:].contiguous()
assert q_pe.shape == (2, 16, 64)
paged_ckv_cache = kv_cache[:, 0].squeeze(2).contiguous()
assert paged_ckv_cache.shape == (2, 16, 512)
paged_kpe_cache = kv_cache[:, 1][..., 0, :64].contiguous()
assert paged_kpe_cache.shape == (2, 16, 64)

decode_output = wrapper.run(
    q_nope=q_nope,
    q_pe=q_pe,
    paged_ckv_cache=paged_ckv_cache,
    paged_kpe_cache=paged_kpe_cache,
)

k_pe_cache = torch.zeros(2,
                         9,
                         64,
                         device=kv_cache.device,
                         dtype=kv_cache.dtype)
k_pe_cache[0, :7, :] = kv_cache[0, 1, :7, 0, :64]
k_pe_cache[1, :9, :] = kv_cache[1, 1, :9, 0, :64]

compressed_kv_normed_cache = torch.zeros(2,
                                         9,
                                         512,
                                         device=kv_cache.device,
                                         dtype=kv_cache.dtype)
compressed_kv_normed_cache[0, :7, :] = kv_cache[0, 0, :7, 0, :]
compressed_kv_normed_cache[1, :9, :] = kv_cache[1, 0, :9, 0, :]

# attn_weights_pe ~ [bsz, 128, kv_len]
attn_weights_pe = torch.matmul(
    q_pe,  # [bsz, num_heads, qk_rope_head_dim]
    k_pe_cache.transpose(
        1, 2),  # [bsz, kv_len, 64] view(bsz, kv_len, self.qk_rope_head_dim)
)
# attn_weights_nope ~ [bsz, 128, kv_len]
attn_weights_nope = torch.matmul(
    q_nope,  # [bsz, 128, 512]
    compressed_kv_normed_cache.transpose(1, 2),  # view(bsz, kv_len, 512)
)

attn_weights = (attn_weights_pe + attn_weights_nope) * 1 / sqrt(512 + 64)

attn_weights_sm = torch.nn.functional.softmax(
    attn_weights,
    dim=-1,
    dtype=torch.float32,
).to(q_nope.dtype)

# attn_output ~ {attn_output.shape}") # [bsz, 128, 512]
attn_output = torch.matmul(
    attn_weights_sm,  # [bsz, 128, kv_len]
    compressed_kv_normed_cache,  # [bsz, kv_len, 512]
)

print(f"{(decode_output - attn_output).abs().sum()=}")

def wmape(target: torch.Tensor, preds: torch.Tensor):
    sum_abs_error = (preds - target).abs().sum().detach().item()
    sum_scale = target.abs().sum().detach().item()
    return sum_abs_error / sum_scale


wmape_value = wmape(decode_output, attn_output)
# assert wmape_value < 0.02, wmape_value
print(f"{wmape_value=}")

cos_similiarity = torch.nn.functional.cosine_similarity(
    decode_output.flatten(), attn_output.flatten(), dim=0)
# assert cos_similiarity > 0.99, cos_similiarity
print(f"{cos_similiarity=}")
