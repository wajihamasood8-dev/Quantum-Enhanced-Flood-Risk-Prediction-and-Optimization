import numpy as np

def evaluate_qubo(x, Q):
    return x.T @ Q @ x


def evaluate_solution(x, Q, ids_array, risk_scores, warning_value,
                      total_costs, flash_flood_locs, interprovincial_locs,
                      ALPHA_VALUE, value_scores, verbose=True):

    selected_indices = np.where(x == 1)[0]
    selected_ids = ids_array[selected_indices].tolist()

    obj_value = evaluate_qubo(x, Q)

    n_selected = int(x.sum())
    n_flash = sum(i in flash_flood_locs for i in selected_indices)
    n_interprov = sum(i in interprovincial_locs for i in selected_indices)

    total_risk_covered = sum(risk_scores[i] for i in selected_indices)
    total_warning_value = sum(warning_value[i] for i in selected_indices)
    total_cost = sum(total_costs[i] for i in selected_indices)

    if verbose:
        print(f"Selected sensors: {selected_ids}")
        print(f"Objective: {obj_value:.2e}")
        print(f"Risk coverage: {total_risk_covered:.2f}")

    return {
        "objective": obj_value,
        "selected": selected_ids,
        "n_selected": n_selected,
        "n_flash": n_flash,
        "n_interprov": n_interprov,
        "risk_coverage": total_risk_covered,
        "warning_value": total_warning_value,
        "cost": total_cost
    }
