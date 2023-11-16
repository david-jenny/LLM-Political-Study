import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pingouin as pg
import networkx as nx
import statsmodels.api as sm
import matplotlib.pyplot as plt
from copy import deepcopy

from plotter_utils import *



# Compute partial correlations pc(X, Y : Z)
def compute_partial_correlations(c_xy, flip_sign=False):
    n = len(c_xy.columns)
    pc_xyz = np.zeros((n, n, n))
    for xi, col_x in enumerate(c_xy.columns):
        for yi, col_y in enumerate(c_xy.columns):
            for zi, col_z in enumerate(c_xy.columns):
                if xi != zi and xi != yi and zi != yi:
                    v_xz = c_xy[col_x][col_z]
                    v_yz = c_xy[col_y][col_z]
                    v_xy = c_xy[col_x][col_y]

                    if flip_sign:
                        # if C(X, Z) < 0 => flip sign of C(X, Z) and C(X, Y)
                        if v_xz < 0:
                            v_xz = -v_xz
                            v_xy = -v_xy
                        
                        # if C(Y, Z) < 0 => flip sign of C(Y, Z) and C(X, Y)
                        if v_yz < 0:
                            v_yz = -v_yz
                            v_xy = -v_xy

                        assert(v_xz >= 0)
                        assert(v_yz >= 0)
                    
                    c_xz_2 = v_xz * v_xz
                    c_yz_2 = v_yz * v_yz
                    pc_xyz[xi, yi, zi] = v_xy - v_xz * v_yz
                    pc_xyz[xi, yi, zi] /= np.sqrt((1 - c_xz_2)*(1 - c_yz_2))

                    # c_ij_2 = c_xy[col_i][col_j] * c_xy[col_i][col_j]
                    # c_kj_2 = c_xy[col_k][col_j] * c_xy[col_k][col_j]
                    # pc_xyz[i, k, j] = c_xy[col_i][col_k] - c_xy[col_i][col_j] * c_xy[col_k][col_j]
                    # pc_xyz[i, k, j] /= np.sqrt((1 - c_ij_2)*(1 - c_kj_2))

    return pc_xyz

# Compute distance metric d(X, Y : Z) = c(X, Y) - pc(X, Y : Z)
def compute_distance_metrics(c_xy, pc_xyz, flip_sign=False):
    n = len(c_xy.columns)
    d_xyz = np.zeros((n, n, n))
    for xi, col_x in enumerate(c_xy.columns):
        for yi, col_y in enumerate(c_xy.columns):
            for zi, col_z in enumerate(c_xy.columns):
                if xi != yi and xi != zi and yi != zi:
                    v_xy = c_xy[col_x][col_y]
                    v_xz = c_xy[col_x][col_z]
                    v_yz = c_xy[col_y][col_z]

                    if flip_sign:
                        # if C(X, Z) < 0 => flip sign of C(X, Z) and C(X, Y)
                        if v_xz < 0:
                            v_xz = -v_xz
                            v_xy = -v_xy
                        
                        # if C(Y, Z) < 0 => flip sign of C(Y, Z) and C(X, Y)
                        if v_yz < 0:
                            v_yz = -v_yz
                            v_xy = -v_xy

                        assert(v_xz >= 0)
                        assert(v_yz >= 0)

                    d_xyz[xi, yi, zi] = v_xy - pc_xyz[xi, yi, zi]

                    # d_xyz[i, k, j] = c_xy[col_i][col_k] - pc_xyz[i, k, j]
    return d_xyz

def compute_node_activity_and_pcpn(c_xy, d_xyz, weight_by_correlation_to_others=False):
    n = len(c_xy.columns)

    d_xz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i==j:
                d_xz[i, j] = 0.
                continue

            if weight_by_correlation_to_others:
                # if two things are very correlated we want to weight them together less than if they are not
                # so we weight them by their uniquness
                # weights is 1 divided by (1 + the average correlation of wi to all others except i and j)
                weights = []
                for wi in range(n):
                    if wi == i or wi == j:
                        weights.append(0.)
                        continue
                    tmp = c_xy.iloc[wi, :]
                    tmp = tmp.drop(c_xy.columns[i])
                    tmp = tmp.drop(c_xy.columns[j])
                    weights.append(1. / (1. + np.mean(np.abs(tmp))))
                weights = np.array(weights) / np.sum(weights)
                # check that all entries >= 0
                assert(np.all(weights >= 0))
                values = d_xyz[i, :, j].copy().flatten()
                d_xz[i, j] = (values * weights).sum()
            else:
                d_xz[i, j] = d_xyz[i, :, j].mean()


    ranking = np.unravel_index(np.argsort(-d_xz, axis=None), d_xz.shape)

    pcpn = nx.DiGraph()

    # add nodes in order of ranking with their respective ranking and name
    node_ranking = []
    for i in ranking[0]:
        if i not in node_ranking:
            node_ranking.append(i)
    for i in ranking[1]:
        if i not in node_ranking:
            node_ranking.append(i)
    for i in range(len(node_ranking)):
        pcpn.add_node(node_ranking[i], rank=i, name=c_xy.columns[node_ranking[i]].replace('_', ' ').replace(' ', '\n'))

    for i in range(3 * (n - 2)):
        pair = ranking[0][i], ranking[1][i]
        weight = d_xz[pair[0], pair[1]]
        pcpn_copy = pcpn.copy()
        pcpn_copy.add_edge(pair[0], pair[1], weight=weight)
        pcpn.add_edge(pair[0], pair[1], weight=weight)

    # turn into dataframe
    d_xz_df = pd.DataFrame(d_xz, columns=c_xy.columns, index=c_xy.columns)
    
    return d_xz_df, ranking, pcpn

def combine_synonyms_and_antonyms(corr, synonym_cutoff=0.9, antonym_cutoff=-0.9):    
    # add all nodes
    G = nx.Graph()
    for key in corr.keys():
        G.add_node(key)
    
    # add all edges
    for key1 in corr.keys():
        for key2 in corr.keys():
            if key1 != key2:
                if corr[key1][key2] >= synonym_cutoff:
                    G.add_edge(key1, key2, weight=corr[key1][key2])
                elif corr[key1][key2] <= antonym_cutoff:
                    G.add_edge(key1, key2, weight=corr[key1][key2])
    
    # get all connected components
    components = list(nx.connected_components(G))

    # compute the partial correlation and distance graph for each variable
    full_pc_xyz = compute_partial_correlations(corr)
    full_d_xyz = compute_distance_metrics(corr, full_pc_xyz)

    combined_d_xyz = np.zeros((len(components), len(components), len(components)))

    all_measurements = []

    for xi, X in enumerate(components):
        for yi, Y in enumerate(components):
            for zi, Z in enumerate(components):
                # for the product of all combinations of X, Y, Z members get
                # the distance graph without the other members
                # then average the distance graph over all members
                measurements = []
                for x in X:
                    for y in Y:
                        for z in Z:
                            measurements.append(full_d_xyz[corr.columns.get_loc(x), corr.columns.get_loc(y), corr.columns.get_loc(z)])
                            # corr = deepcopy(corr)
                            # drop all columns in X, Y, Z except for x, y, z
                            # for key in corr.keys():
                            #     if key in X or key in Y or key in Z:
                            #         if key != x and key != y and key != z:
                                        # corr = corr.drop(key, axis=1)
                                        # corr = corr.drop(key, axis=0)
                            # pc_xyz = compute_partial_correlations(corr)
                            # d_xyz = compute_distance_metrics(corr, pc_xyz)
                            # measurements.append(d_xyz[corr.columns.get_loc(x), corr.columns.get_loc(y), corr.columns.get_loc(z)])

                measurements = np.abs(measurements)
                combined_d_xyz[xi, yi, zi] = np.mean(measurements)

                all_measurements.append((X, Y, Z, measurements, np.var(measurements)))

                # print(f'Variance of measurements for {X}, {Y}, {Z}: {measurements}')

                # average over all measurements
                # combined_d_xyz[xi, yi, zi] = np.mean(measurements)[xi, yi, zi]

def compute_distance_matrix_from_df(df, optimize_corr=True, flip_sign=False, weight_by_correlation_to_others=False):
    c_xy = df.corr()

    # make it so that there are mostly positive correlations => if average correlation is negative, flip sign of all correlations
    if optimize_corr:
        change = True
        while change:
            change=False
            for col in c_xy.columns:
                if c_xy[col].sum() - c_xy[col][col] < 1e-8:
                    c_xy[col] = -c_xy[col]
                    c_xy.loc[col] = -c_xy.loc[col]
                    change=True

    pc_xyz = compute_partial_correlations(c_xy, flip_sign=flip_sign)
    d_xyz = compute_distance_metrics(c_xy, pc_xyz, flip_sign=flip_sign)
    d_xz, ranking, pcp_graph = compute_node_activity_and_pcpn(c_xy, d_xyz, weight_by_correlation_to_others=weight_by_correlation_to_others)

    return d_xz, ranking, pcp_graph

def fit_linear_model(X, Y):
    model = LinearRegression()

    model.fit(X, Y)
    return model

def fit_nonlinear_model(X, Y):
    # model = sm.OLS(Y, X)
    model = SVR(kernel='rbf')
    results = model.fit(X, Y)
    return results

def remove_Y_from_X(X, Y, fit_fn):
    new_X = X.copy()
    Y_ = Y.copy().values.reshape(-1, 1)
    for x in X.columns:
        model = fit_fn(Y_, X[x])
        new_X[x] = X[x] - model.predict(Y_)
    return new_X

def predict_Y_from_X(X, Y, fit_fn):
    model = fit_fn(X, Y)
    return model.predict(X)

def error_of_Y_from_X(X, Y, fit_fn):
    model = fit_fn(X, Y)
    return np.sqrt(np.mean((model.predict(X) - Y)**2))

def accuracy_of_Y_from_X(X, Y, fit_fn):
    X = X.copy()
    Y = Y.copy()

    # normalize
    for col in X.columns:
        std = X[col].std()

        if abs(std) < 1e-8:
            X[col] = 0.
        else:
            mean = X[col].mean()
            X[col] = (X[col] - mean) / std
    std = Y.std()
    if abs(std) < 1e-8:
        Y = 0.
    else:
        mean = Y.mean()
        Y = (Y - mean) / std

    model = fit_fn(X, Y)
    return 1.0 - np.sqrt(np.mean((model.predict(X) - Y)**2))

def error_of_Y_as_function_of_X_and_X_star(X, Y, fit_fn_remove, fit_fn_predict):
    model = fit_fn_predict(X, Y)
    error = np.sqrt(np.mean((model.predict(X) - Y)**2))

    X_star = remove_Y_from_X(X, Y, fit_fn_remove)

    # print(X_star.corr())
    tmp = X_star.copy()
    tmp['RES'] = Y
    print(tmp.corr())

    model_star = fit_fn_predict(X_star, Y)
    error_star = np.sqrt(np.mean((model_star.predict(X_star) - Y)**2))

    return error, error_star

# removes each variable y in X from all others once and then predicting x* from X* as function of the remaining X*
# returns accuracy_matrix df where am[zk][yk] = accuracy of prediction zk as function of others when removing yk from all
# if star is False it does not actually remove y from the others
def prediction_accuracy_matrix(X, fit_fn_remove, fit_fn_predict, star=True):
    matrix = pd.DataFrame(columns=X.columns, index=X.columns, dtype=np.float64)

    for yk in X.columns:
        print(f'yk={yk}')
        matrix[yk][yk] = 0.
        # create df without y and then remove y from it
        X_star = X.copy()
        X_star = X_star.drop(yk, axis=1)
        if star:
            X_star = remove_Y_from_X(X_star, X[yk], fit_fn_remove)

        # normalize all columns
        for ck in X_star.columns:
            std = X_star[ck].std()

            if abs(std) < 1e-8:
                X_star[ck] = 0.
            else:
                mean = X_star[ck].mean()
                X_star[ck] = (X_star[ck] - mean) / std
        
        # predict members of X* as function of others
        for zk in X_star.columns:
            xx = X_star[[k for k in X_star.columns if k!=zk]]
            yy = X_star[zk]

            pred = predict_Y_from_X(xx, yy, fit_fn_predict)
            err = np.sqrt(np.mean((pred - yy)**2))
            matrix[zk][yk] = 1.0 - err
    
    return matrix

def compute_dependency_set_ranking(XORG, Y, fit_fn_predict, error_ratio_threshold=0.9):
    """
    We want to find the ordered list of minimal subsets of X that best fit Y, ordered by their error.

    This is achieved by starting with the complete set and removing parts in a tree like fashion.
    For performance pruning can be used.
    """

    # queue of subsets to process, sorted so that the first has the smallest error
    import queue
    q = queue.PriorityQueue()

    best_error = error_of_Y_from_X(XORG, Y, fit_fn_predict)
    q.put((best_error, list(XORG.columns)))

    found_subsets = []

    # while there are subsets to process, process them
    while not q.empty():
        # get the subset with the smallest error
        error, subset = q.get()
        if best_error / error < error_ratio_threshold:
            # if the error is too large, stop
            break

        # add to found
        found_subsets.append((error, subset))

        # only allow the removal of elements up to the first element that is not in the dependency set
        # this allows avoiding looking at the same dependency set multiple times in different branches
        # of the tree
        new_subsets = []
        for i in range(len(subset)):
            if XORG.columns[i] != subset[i] or len(subset) == 1:
                break
            ns = [k for k in subset if k != subset[i]]
            # print(type(XORG[ns]))
            # print(XORG[ns].shape)
            # print(XORG.dtypes)
            error = error_of_Y_from_X(XORG[ns], Y, fit_fn_predict)
            new_subsets.append((error, ns))
        
        # TODO: prune and add threshold checks

        # add the new subsets to the queue
        for error, ns in new_subsets:
            q.put((error, ns))
    
    # sort the found subsets by error
    found_subsets.sort(key=lambda x: x[0])

    return found_subsets

def compute_dependency_set_ranking_from_df(df, fit_fn_predict, error_ratio_threshold=0.5):
    combined_ranking = []

    df_norm = df.copy()
    for col in df_norm.columns:
        mean = df_norm[col].mean()
        std = df_norm[col].std()
        if abs(std) < 1e-8:
            df_norm[col] = 0.
        else:
            df_norm[col] = (df_norm[col] - mean) / std

    for col in df.columns:
        XX = df_norm[[k for k in df_norm.columns if k != col]]
        YY = df_norm[col]

        ranking = compute_dependency_set_ranking(XX, YY, fit_fn_predict, error_ratio_threshold=error_ratio_threshold)
        combined_ranking = combined_ranking + [(error, col, subset) for error, subset in ranking]
    
    combined_ranking.sort(key=lambda x: x[0])

    return combined_ranking

# insanely inefficient, implement the better version...
def compute_possible_connection_set(XORG, Y, fit_fn_predict, subset_size_depth=2, acc_add_unknown_threshold=0.025):
    U = set(XORG.columns)
    K = set()

    K_debug = [[] for i in range(subset_size_depth)]

    # for all subsets of size i <= subset_size_depth in U and between U and K check if the unknown part helps
    i = 1
    while i <= subset_size_depth:
        found = False

        # get subsets of U of up to size i
        for u_n in range(1, i+1):
            u_c = itertools.combinations(U, u_n)
            for u_s in u_c:
                u_s = set(u_s)
                # get subsets of K of up to size i-u_n
                for k_n in range(0, i+1-u_n):
                    if k_n == 0:
                        k_c = [()]
                    else:
                        k_c = itertools.combinations(K, k_n)
                    for k_s in k_c:
                        k_s = set(k_s)
                        # check if prediction of Y from XORG[k_s + u_s] is better than from XORG[k_s]
                        pred_k_u = accuracy_of_Y_from_X(XORG[list(k_s.union(u_s))], Y, fit_fn_predict)

                        if len(k_s) == 0:
                            pred_k = 0.
                        else:
                            pred_k = accuracy_of_Y_from_X(XORG[list(k_s)], Y, fit_fn_predict)

                        if pred_k_u - pred_k > acc_add_unknown_threshold:
                            # if yes, add the unknown part to K
                            K = K.union(u_s)
                            U = U - u_s

                            K_debug[i-1].append((pred_k_u, pred_k, u_s, k_s))
                            
                            found = True
                            break
                    if found: break
                if found: break
            if found:break
        if not found:
            i += 1

    return K, K_debug

def compute_possible_connection_matrix(XORG, fit_fn_predict, subset_size_depth=2, acc_add_unknown_threshold=0.025):
    matrix = pd.DataFrame(columns=XORG.columns, index=XORG.columns, dtype=np.int8)

    for col in XORG.columns:
        print(f'col={col}')
        matrix[col][col] = 0.
        # create df without y and then remove y from it
        XX = XORG[[k for k in XORG.columns if k != col]].copy()
        YY = XORG[col].copy()

        possible_connections, _ = compute_possible_connection_set(XX, YY, fit_fn_predict, subset_size_depth=subset_size_depth, acc_add_unknown_threshold=acc_add_unknown_threshold)
        
        for zk in XORG.columns:
            if zk in possible_connections:
                matrix[col][zk] = 1
            else:
                matrix[col][zk] = 0
    
    return matrix

def compute_information_flow(XORG, possible_connection_matrix, X, Y, fit_fn_predict):
    """
    We check if the prediction of B from it's dependency set is improved if we add the residue of A given B
    Aka if pred(A|dependency_set(A) + residue) is better than pred(A|dependency_set(A)) then we have unexplained information flow from A to B through the residue.
    Meaning that the residue contains information about A that is not contained in the dependency set of A. and that B causes A and not the other way around.
    """

    print(f'{X} -> {Y}')

    Y_dependency_set = possible_connection_matrix.columns[possible_connection_matrix.loc[Y] == 1]
    print(f'Y_dependency_set={Y_dependency_set}')

    if len(Y_dependency_set) == 0:
        return 0., 0., 0.

    YD = XORG[Y_dependency_set].copy()
    YY = XORG[Y].copy()
    YY_in = XORG[[Y]].copy()

    acc_Y_from_YD = accuracy_of_Y_from_X(YD, YY, fit_fn_predict)

    XX = XORG[X].copy()

    X_residue_Y = XX - predict_Y_from_X(YY_in, XX, fit_fn_predict)

    # print(XX.std(), X_residue_Y.std())
    print(f'X std: {XX.std()}, X_residue_Y std: {X_residue_Y.std()}')
    # print correlation of X, Y, X_residue_Y
    tmp = XORG[[X, Y]].copy()
    tmp['RES'] = X_residue_Y
    print(tmp.corr())

    XRES = YD.copy()
    XRES['RES'] = X_residue_Y

    acc_Y_from_YD_and_RES = accuracy_of_Y_from_X(XRES, YY, fit_fn_predict)

    print(f'acc_Y_from_YD={acc_Y_from_YD}, acc_Y_from_YD_and_RES={acc_Y_from_YD_and_RES}, acc_Y_from_only_X={accuracy_of_Y_from_X(XORG[[X]], YY, fit_fn_predict)}, acc_Y_from_only_RES={accuracy_of_Y_from_X(pd.DataFrame.from_dict({"RES": X_residue_Y}), YY, fit_fn_predict)}')

    return acc_Y_from_YD_and_RES - acc_Y_from_YD, acc_Y_from_YD_and_RES, acc_Y_from_YD

def compute_information_flow_matrix(XORG, possible_connection_matrix, fit_fn_predict):
    matrix = pd.DataFrame(columns=XORG.columns, index=XORG.columns, dtype=np.float64)
    matrix_all_and_residue = pd.DataFrame(columns=XORG.columns, index=XORG.columns, dtype=np.float64)
    matrix_all = pd.DataFrame(columns=XORG.columns, index=XORG.columns, dtype=np.float64)

    for col in XORG.columns:
        print(f'col={col}')
        matrix[col][col] = 0.
        # create df without y and then remove y from it
        XX = XORG[[k for k in XORG.columns if k != col]].copy()
        YY = XORG[col].copy()

        for zk in XORG.columns:
            if zk != col:
                # matrix[col][zk], _, _ = compute_information_flow(XORG, possible_connection_matrix, col, zk, fit_fn_predict)
                matrix[col][zk], matrix_all_and_residue[col][zk], matrix_all[col][zk] = compute_information_flow(XORG, possible_connection_matrix, col, zk, fit_fn_predict)

    return matrix, matrix_all_and_residue, matrix_all

def plot_dag(g, fig_size=(8, 8)):
    # plot graph as hierarchy, make edges thiker and less transparent the higher the weight
    # pos = nx.nx_pydot.graphviz_layout(g, prog='dot')
    # pos = nx_agraph.graphviz_layout
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # for u, v, d in g.edges(data=True):
    #     ax.annotate("",
    #                 xy=pos[v], xycoords='data',
    #                 xytext=pos[u], textcoords='data',
    #                 arrowprops=dict(arrowstyle="->", color='black', alpha=max(0.1, min(1, d['weight'])), linewidth=max(0.1, min(10, d['weight']*10))))

    # # add legend for the edge colors
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min([d["weight"] for (u, v, d) in g.edges(data=True)]), vmax=max([d["weight"] for (u, v, d) in g.edges(data=True)])),
    )

    plt.colorbar(sm)
        
    nx.draw_networkx_nodes(g, pos, node_size=1000, alpha=0.5, node_color='blue', ax=ax)
    nx.draw_networkx_edges(g, pos, alpha=0.5, ax=ax,
                        width=[(max(min(d["weight"], 0.15), 0.1)-0.09) * 100 for (u, v, d) in g.edges(data=True)],
                        edge_color=[d["weight"] for (u, v, d) in g.edges(data=True)],
                        arrowsize=10,
                        arrowstyle="-|>")
    nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)

    return fig, ax

def create_directed_asyclic_graph(dis_matrix, fig_size=(8, 8)):
    # given a matrix of connection_strengths between nodes, sort the edges by strength and add them to the graph if they don't create a cycle
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    residual_matrix = dis_matrix.copy()

    # create graph
    g = nx.DiGraph()

    for key in dis_matrix.columns:
        g.add_node(key) #, name=key)
    
    edges = list(dis_matrix.stack().reset_index().values)
    edges.sort(key=lambda x: x[2], reverse=True)

    # add edges
    for u, v, d in edges:
        if d<=0:
            break
        if not nx.has_path(g, v, u):
            g.add_edge(u, v, weight=d)
            # print(f"Adding edge {u} -> {v}: {d}")
            assert(d==residual_matrix[v][u])
            residual_matrix[v][u] = 0.
        # else:
        #     print(f"Can't add back edge {u} -> {v}: {d}")

    # create layout hierarchy
    for node in g.nodes():
        g.nodes[node]['rank'] = 0
    for node in g.nodes():
        for other_node in g.nodes():
            if nx.has_path(g, node, other_node):
                g.nodes[node]['rank'] += 1

    fig, ax = plot_dag(g, fig_size=fig_size)

    return g, residual_matrix, fig, ax

def dependencies_to_definition(matrix, target, max_depth=3):
    # we want to find the definition of the target variable, we do this by checking what the maximum flow from variabel to target is to get it's weight in the definition

    # create graph
    g = nx.DiGraph()

    for key in matrix.columns:
        g.add_node(key)

    edges = list(matrix.stack().reset_index().values)
    edges.sort(key=lambda x: x[2], reverse=True)

    # add edges
    for u, v, d in edges:
        if d<=0:
            break
        g.add_edge(u, v, weight=d)

    # set the capacity of all edges to the weight
    for u, v, d in g.edges(data=True):
        d['capacity'] = d['weight']
    
    definition = []
    accumulated = []
    accumulated_definition = {}

    for max_steps in range(1, min(max_depth + 1, len(matrix.columns))):
        definition.append({})

        for key in matrix.columns:
            if key == target:
                continue
            
            accumulated_flow = 0
            # check all paths from i to target that have a length of max_steps and take the min weight along the path
            for path in nx.all_simple_paths(g, key, target, cutoff=max_steps):
                # check if path is exactly max_steps long
                if len(path) != max_steps+1:
                    continue
                accumulated_flow += min([g[u][v]['weight'] for u, v in zip(path[:-1], path[1:])])
                # take multiplication instead of min
                # accumulated_flow += np.prod([g[u][v]['weight'] for u, v in zip(path[:-1], path[1:])])
            definition[-1][key] = accumulated_flow


        # accumulated_definition = {k: accumulated_definition[k] + definition[-1][k] for k in accumulated_definition.keys()} if len(accumulated_definition) > 0 else definition[-1].copy()
        # actually decrease value based on distance
        accumulated_definition = {k: accumulated_definition[k] + definition[-1][k] / max_steps**3 for k in accumulated_definition.keys()} if len(accumulated_definition) > 0 else definition[-1].copy()

        definition[-1] = list(sorted(definition[-1].items(), key=lambda x: x[1], reverse=True))
        accumulated.append(list(sorted(accumulated_definition.items(), key=lambda x: x[1], reverse=True)))

    # remove all edges with negative weight
    edges = list(g.edges(data=True))
    for u, v, d in edges:
        if d['weight'] <= 0:
            g.remove_edge(u, v)
    
    max_flow_definition = {}
    # for all variables, check the maximum flow to the target
    for key in matrix.columns:
        if key == target:
            continue
        max_flow_definition[key] = nx.maximum_flow_value(g, key, target)
    
    max_flow_definition = list(sorted(max_flow_definition.items(), key=lambda x: x[1], reverse=True))
    
    return definition, accumulated, max_flow_definition
