from numpy.core.multiarray import einsum as c_einsum
from numpy.core.numeric import asarray, asanyarray, result_type

__all__ = ['einsum']


einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

def _compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : int
        The resulting product.
    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets.

    Paramaters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction
    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)
    return (new_result, remaining, idx_removed, idx_contract)


def _path_optimal(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path.

    Paramaters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The optimal order of pair contractions.
    """

    current = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        new = []

        # Grab all unique pairs
        comb_iter = []
        for x in range(len(input_sets) - iteration):
            for y in range(x + 1, len(input_sets) - iteration):
                comb_iter.append((x, y))

        for curr in current:
            cost, positions, remaining = curr
            for con in comb_iter:

                contract = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = contract

                # Sieve the results based on memory_limit
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Find cost
                new_cost = _compute_size_by_dict(idx_contract, idx_dict)
                if len(idx_removed) > 0:
                    new_cost *= 2

                # Build (total_cost, positions, indices_remaining)
                new_cost += cost
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_input_sets))

        # Update list to iterate over
        current = new

    # If we have not found anything return single einsum contraction
    if len(new) == 0:
        return [tuple(range(len(input_sets)))]

    new.sort()
    path = new[0][1]
    return path


def _path_greedy(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the best pair contraction at each iteration. The best pair is found
    by minimizing the tuple ``(-prod(indices_removed), cost)``.  Another way to say
    this is it tries to remove the largest index at the lowest cost.  What this
    amounts to is prioritizing matrix multiplication or inner product operations,
    then Hadamard like operations, and finally outer operations. Outer products are
    limited by ``memory_limit``.

    Paramaters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The greedy order of pair contractions.
    """

    path = []
    for iteration in range(len(input_sets) - 1):
        iteration_results = []
        comb_iter = []

        # Grab all unique pairs
        for x in range(len(input_sets)):
            for y in range(x + 1, len(input_sets)):
                comb_iter.append((x, y))

        for positions in comb_iter:

            contract = _find_contraction(positions, input_sets, output_set)
            idx_result, new_input_sets, idx_removed, idx_contract = contract

            # Sieve the results based on memory_limit
            if _compute_size_by_dict(idx_result, idx_dict) > memory_limit:
                continue

            # Build sort tuple
            removed_size = _compute_size_by_dict(idx_removed, idx_dict)
            cost = _compute_size_by_dict(idx_contract, idx_dict)
            sort = (-removed_size, cost)

            # Add contraction to possible choices
            iteration_results.append([sort, positions, new_input_sets])

        # If we did not find a new contraction contract remaining
        if len(iteration_results) == 0:
            path.append(tuple(range(len(input_sets))))
            break

        # Sort based on first index
        iteration_results.sort()
        best = iteration_results[0]
        path.append(best[1])
        input_sets = best[2]

    return path


# Rewrite einsum to handle different cases
def einsum(*operands, **kwargs):
    """
    Evaluates the Einstein summation convention based on the operands,
    differs from einsum by utilizing intermediate arrays to
    reduce overall computational time.

    Produces results identical to that of the einsum function; however,
    the contract function expands on the einsum function by building
    intermediate arrays to reduce the computational scaling and utilizes
    BLAS calls when possible.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    tensordot : bool, optional (default: True)
        If true use tensordot where possible.
    path : bool or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' means a N^3 algorithm that greedyally
            chooses the best algorithm.
        - 'optimal' means a N! algorithm that tries all possible ways of
            contracting the listed tensors.

    memory_limit : int, optional (default: largest input or output array size)
        Maximum number of elements allowed in an intermediate array.
    return_path : bool, optional (default: False)
        If true retuns the path and a string representation of the path.


    Returns
    -------
    if return_path:
        path : list
            The order of contracted indices.
        path_string : string
            A string representation of the contraction.

    else:
        output : ndarray
            The results based on Einstein summation convention.

    See Also
    --------
    einsum, tensordot, dot

    Notes
    -----
    Subscript labels follow the same convention as einsum with the exception
    that integer indexing and ellipses are not currently supported.

    The amount of extra memory used by this function depends greatly on the
    einsum expression and BLAS usage.  Without BLAS the maximum memory used is:
    ``(number_of_terms / 2) * memory_limit``.  With BLAS the maximum memory used
    is: ``max((number_of_terms / 2), 2) * memory_limit``.  For most operations
    the memory usage is approximately equivalent to the memory_limit.

    Note: BLAS is not yet implemented.
    One operand operations are supported by calling ``np.einsum``.  Two operand
    operations are first checked to see if a BLAS call can be utilized then
    defaulted to einsum.  For example ``np.contract('ab,bc->', a, b)`` and
    ``np.contract('ab,cb->', a, b)`` are prototypical matrix matrix multiplication
    examples.  Higher dimensional matrix matrix multiplicaitons are also considered
    such as ``np.contract('abcd,cdef', a, b)`` and ``np.contract('abcd,cefd', a,
    b)``.  For the former, GEMM can be called without copying data; however, the
    latter requires a copy of the second operand.  For all matrix matrix
    multiplication examples it is beneficial to copy the data and call GEMM;
    however, for matrix vector multiplication it is not beneficial to do so.  For
    example ``np.contract('abcd,cd', a, b)`` will call GEMV while
    ``np.contract('abcd,ad', a, b)`` will call einsum as copying the first operand
    then calling GEMV does not provide a speed up compared to calling einsum.

    For three or more operands contract computes the optimal order of two and
    one operand operations.  The ``optimal`` path scales like N! where N is the
    number of terms and is found by calculating the cost of every possible path and
    choosing the lowest cost.  This path can be more costly to compute than the
    contraction itself for a large number of terms (~N>7).  The ``greedy``
    path scales like N^3 and first tries to do any matrix matrix multiplications,
    then inner products, and finally outer products.  This path usually takes a
    trivial amount of time to compute unless the number of terms is extremely large
    (~N>20).  The greedy path typically computes the most optimal path, but
    is not guaranteed to do so.  Both of these algorithms are sieved by the
    variable memory to prevent the formation of very large tensors.

    Examples
    --------
    A index transformation example, contract runs ~2000 times faster than
    einsum even for this small example.

    Note: BLAS will be True for all contractions here when everything is
    finished.

    >>> from numpy.linalg import contract
    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> opt_path = contract('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, return_path=True)

    >>> print(opt_path[0])
    [(2, 0), (3, 0), (2, 0), (1, 0)]
    >>> print(opt_path[1])
    Complete contraction:  ea,fb,abcd,gc,hd->efgh
           Naive scaling:   8
    --------------------------------------------------------------------------------
    scaling   BLAS                  current                                remaining
    --------------------------------------------------------------------------------
       5     False            abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5     False            bcde,fb->cdef                         gc,hd,cdef->efgh
       5     False            cdef,gc->defg                            hd,defg->efgh
       5     False            defg,hd->efgh                               efgh->efgh

    >>> opt_result = contract('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, path=opt_path[0])
    >>> ein_result = np.einsum('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)
    >>> np.allclose(ein_result, opt_result)
    True
    """

    # Grab non-einsum kwargs
    optimize_arg = kwargs.pop('optimize', False)
    if optimize_arg is True:
        optimize_arg = 'greedy'
    return_path_arg = kwargs.pop("return_path", False)

    # If no optimization run pure einsum
    if (optimize_arg is False) and (return_path_arg is False):
        return c_einsum(*operands, **kwargs)

    # Special handeling if out is specified
    specified_out = False
    out_array = kwargs.pop('out', None)
    if out_array is not None:
        specified_out = True
    

    ### Start python side parsing
    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [asanyarray(v) for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain " \
                                    "either int or Ellipsis")
            if num != last:
                subscripts += ","
 
        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain " \
                                    "either int or Ellipsis")
    # Checkout for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",","").replace("->","")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                ellipse_count = max(len(operands[num].shape), 1) - (len(sub) - 3)
                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    split_subscripts[num] = sub.replace('...', ellipse_inds[-ellipse_count:])

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses 
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input" % char)

    ### Finished python side parsing

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Make sure number operands is equivalent to the number of terms
    if len(input_list) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "\
                          "number of operands.")

    # Get length of each unique dimension and ensure all dimension are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "\
              "correct number of indices for operand %d.", input_subscripts[tnum], tnum)
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                if dimension_dict[char] != dim:
                    raise ValueError("Size of label '%s' for operand %d does "\
                                     "not match previous terms.", char, tnum)
            else:
                dimension_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_subscript]:
        size_list.append(_compute_size_by_dict(term, dimension_dict))
    out_size = max(size_list)
    memory_arg = out_size

    # If no rank reduction or only one input leave it to einsum
    if ((indices == output_set) or (len(input_list) == 1)) and not return_path_arg:
        if specified_out:
            kwargs["out"] = out_array
        return einsum(subscripts, *operands, **kwargs)

    optimize_arg = "greedy"
    # Compute path
    if not isinstance(optimize_arg, str):
        path = optimize_arg
    elif len(input_list) == 1:
        path = [(0)]
    elif len(input_list) == 2:
        path = [(0, 1)]
    elif optimize_arg == "greedy":
        # Maximum memory should be at most out_size for this algorithm
        memory_arg = min(memory_arg, out_size)
        path = _path_greedy(input_sets, output_set, dimension_dict, memory_arg)
    elif optimize_arg == "optimal":
        path = _path_optimal(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", optimize_arg)

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    contraction_list = []
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
            last = True
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])
            last = False

        input_list.append(idx_result)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result
        contraction = (contract_inds, False, einsum_str, input_list[:], last)
        contraction_list.append(contraction)

    # Return the path along with a nice string representation
    if return_path_arg:
        overall_contraction = input_subscripts + "->" + output_subscript
        header = ("scaling", "BLAS", "current", "remaining")

        path_print = "Complete contraction:  %s\n" % overall_contraction
        path_print += "       Naive scaling:%4d\n" % len(indices)
        path_print += "-" * 80 + "\n"
        path_print += "%6s %6s %24s %40s\n" % header
        path_print += "-" * 80 + "\n"

        path = []
        for inds, gemm, einsum_str, remaining, last in contraction_list:
            remaining_str = ",".join(remaining) + "->" + output_subscript
            path_run = (len(idx_contract), gemm, einsum_str, remaining_str)
            path_print += "%4d    %6s %24s %40s\n" % path_run
            path.append(inds)

        return (path, path_print)

    # Start contraction loop
    for inds, gemm, einsum_str, remaining, last in contraction_list:
        tmp_operands = []
        for x in inds:
            tmp_operands.append(operands.pop(x))
        
        # If out was specified
        if last and specified_out:
            kwargs["out"] = out_array

        # Do the contraction
        new_view = einsum(einsum_str, *tmp_operands, **kwargs)

        # Append new items and derefernce what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return None
    else:
        return operands[0]
