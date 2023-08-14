import DistGNN
import torch
import dgs

indptr = torch.tensor([0, 4, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10]).cuda()
indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda()
probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]).cuda()

cache_nids = torch.tensor([0, 1, 5]).cuda()

sub_indptr = dgs.ops._Test_ExtractIndptr(cache_nids, indptr)
sub_indices = dgs.ops._Test_ExtractEdgeData(cache_nids, indptr, sub_indptr,
                                            indices)
sub_probs = dgs.ops._Test_ExtractEdgeData(cache_nids, indptr, sub_indptr,
                                          probs)
print(sub_indptr)
print(sub_indices)
print(sub_probs)