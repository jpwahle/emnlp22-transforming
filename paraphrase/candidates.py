def choose_pareto_optimal_candidate(
    original_paragraph, paraphrased_candidates, metrics
):
    if len(paraphrased_candidates) == 1:
        return paraphrased_candidates[0]
    # Find the pareto optimal candidate according to a list of metrics
    pareto_optimal_candidate = None
    pareto_optimal_candidate_score = 0
    for candidate in paraphrased_candidates:
        candidate_score = 0
        scores = []
        for metric in metrics:
            score = metric(original_paragraph, candidate)
            scores.append(score)
        # Weight all four metrics equally
        candidate_score = sum(scores) / len(scores)
        if candidate_score > pareto_optimal_candidate_score:
            pareto_optimal_candidate = candidate
            pareto_optimal_candidate_score = candidate_score

    return pareto_optimal_candidate
