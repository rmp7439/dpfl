# Stage 1 Learning Note: Dirichlet Alpha Parameter

This note documents our observations from completing the Dirichlet Non-IID split (Stage 1).

## What `alpha` controls
In the context of federated learning data partitioning, we use a Dirichlet distribution to assign samples of each class to different clients. The parameter `alpha` (concentration parameter) controls the **heterogeneity** (non-IIDness) of the resulting client datasets. Specifically, it dictates how the proportions of a particular class are distributed across clients.

## Why `alpha=0.1` creates strong heterogeneity
When `alpha < 1` (e.g., `alpha = 0.1`), the Dirichlet distribution pushes the probability mass toward the extremes. This means for any given class, the generated proportions will heavily favor one or two clients, leaving the other clients with near-zero proportions of that class. As a result, each client receives a highly unbalanced subset of classes, perfectly simulating a severely non-IID environment.

## What `alpha=1.0` would generally do
When `alpha = 1.0`, the Dirichlet distribution becomes equivalent to a uniform distribution over the simplex. This means every possible valid split of class proportions across clients is equally likely. It creates moderate heterogeneity—clients will have different amounts of each class, but it usually isn't as extreme or polarized as `alpha=0.1`.

## What `alpha=10` would generally do
When `alpha > 1` (e.g., `alpha = 10`), the distribution pushes the probability mass toward the center (the uniform split). If `alpha` is very large, the proportions assigned to each client will be nearly equal (e.g., ~33% for each of 3 clients). This results in a nearly **IID** (independent and identically distributed) split, where every client has roughly the same class distribution as the global dataset.
