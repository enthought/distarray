class ClientMap(object):
    '''
    Governs the mapping between global indices and process ranks.

    Works with the LocalMap classes to facilitate communication between global
    and local processes.

    Invariants: TODO

    '''

    # Does proc-grid data need to be in here?

    # How does the DAP metadata need to be represented on client-side?

    # Need to hold on to mapping between proc_grid_rank and global indices.

    # Need to hold on to mapping between process rank and proc_grid tuple.

    def __init__(self, global_indices, proc_grid_ranks):
        self.global_indices = global_indices
        self.proc_grid_ranks = proc_grid_ranks

    def possibly_owning_ranks(self, idx):
        pass
