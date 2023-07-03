def set_significance(p):
    if p < 0.05:
        significance = '*'
    elif p < 0.005:
        significance = '**'
    else:
        significance = 'NA'
    return significance
