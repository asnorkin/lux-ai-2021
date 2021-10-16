# contains designed heuristics
# which could be fine tuned

import builtins as __builtin__

import numpy as np

from lux.game import Game, Unit
from lux.game_position import Position


def find_best_cluster(game_state: Game, unit: Unit, distance_multiplier = -0.5, DEBUG=False):
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    # passing game_state attributes to compute travel range
    unit.compute_travel_range(game_state)

    # for debugging
    score_matrix_wrt_pos = game_state.init_matrix()

    # default response is not to move
    best_position = unit.pos
    best_cell_value = (0,0,0,0)

    # calculate how resource tiles and how many units on the current cluster
    current_leader = game_state.xy_to_resource_group_id.find(tuple(unit.pos))
    units_mining_on_current_cluster = len(game_state.units_mining_on_cluster[current_leader])
    resource_size_of_current_cluster = game_state.xy_to_resource_group_id.get_point(current_leader)

    # only consider other cluster if the current cluster has at least one agent mining
    consider_different_cluster = units_mining_on_current_cluster > 0

    # must consider other cluster if the current cluster has more agent than tiles
    consider_different_cluster_must = units_mining_on_current_cluster >= resource_size_of_current_cluster

    def _collection_rate(leader):
        if leader in game_state.wood_exist_xy_set:
            return game_state.wood_collection_rate
        if leader in game_state.coal_exist_xy_set:
            return game_state.coal_collection_rate
        if leader in game_state.uranium_exist_xy_set:
            return game_state.uranium_collection_rate
        return 0

    for y in game_state.y_iteration_order:
        for x in game_state.x_iteration_order:

            # what not to target
            if (x,y) in game_state.targeted_xy_set:
                continue
            if (x,y) in game_state.targeted_for_building_xy_set:
                continue
            if (x,y) in game_state.opponent_city_tile_xy_set:
                continue
            if (x,y) in game_state.player_city_tile_xy_set:
                continue

            # No resources => do not consider
            if game_state.convolved_collectable_tiles_matrix[y, x] == 0:
                continue

            target_leader = game_state.xy_to_resource_group_id.find((x, y))

            # Do not consider tiles outside clusters
            if target_leader is None:
                continue

            # using path distance
            distance = game_state.retrieve_distance(unit.pos.x, unit.pos.y, x, y)
            distance = max(0.5, distance)  # prevent zero error

            # Target bonus
            target_bonus = 1
            if consider_different_cluster:
                units_on_cluster = len(game_state.units_locating_or_targeting_on_cluster[target_leader])
                if units_on_cluster == 0:
                    point = game_state.xy_to_resource_group_id.get_point(target_leader)
                    size = game_state.xy_to_resource_group_id.get_size(target_leader)
                    collection_rate = _collection_rate(target_leader)
                    avg_resource_amount = game_state.cluster_resource_amounts[target_leader] / size
                    target_bonus = point * collection_rate * avg_resource_amount / np.sqrt(distance)

                if consider_different_cluster_must:
                    target_bonus *= 1000

            elif target_leader == current_leader:
                target_bonus = 1000000

            # prefer empty tile because you can build afterwards quickly
            empty_tile_bonus = 1/(0.5+game_state.distance_from_buildable_tile[y,x])

            # no empty tile preference if resource is not wood
            for dx,dy in game_state.dirs_dxdy:
                xx, yy = x+dx, y+dy
                if (xx,yy) in game_state.wood_exist_xy_set:
                    break
            else:
                empty_tile_bonus = 1/(0.5+max(1,game_state.distance_from_buildable_tile[y,x]))

            # estimate target score
            if distance <= unit.travel_range:
                cell_value = (target_bonus,
                              empty_tile_bonus * game_state.convolved_collectable_tiles_matrix[y,x] * distance ** distance_multiplier,
                              game_state.distance_from_edge[y,x],
                              -game_state.distance_from_opponent_assets[y,x])
                score_matrix_wrt_pos[y,x] = cell_value[0]*1000 + cell_value[1]*100 + cell_value[2]*10 + cell_value[3]

                # update best target
                if cell_value > best_cell_value:
                    best_cell_value = cell_value
                    best_position = Position(x,y)

    # for debugging
    game_state.heuristics_from_positions[tuple(unit.pos)] = score_matrix_wrt_pos

    return best_position, best_cell_value
