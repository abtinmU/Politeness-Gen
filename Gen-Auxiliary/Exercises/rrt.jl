using Random

struct PlannerParams
    rrt_iters::Float64
    rrt_dt::Float64
    refine_iters::Float64
    refine_std::Float64
end

function sampleAPoint(scene::Scene)
    x = rand(Uniform(scene.xmin, scene.xmax))
    y = rand(Uniform(scene.ymin, scene.ymax))
    return Point(x, y)
end

function isInObstacle(scene::Scene, sample::Point)
    for obs in scene.obstacles
        for vertex in obs.vertices
            if scene.xmin <= vertex.x <= scene.xmax && scene.ymin <= vertex.y <= scene.ymax
                return true
            end
        end
    end
    return false
end

function distance(point1::Point, point2::Point)
    return sqrt((point1.x - point2.x)^2 + (point1.y - point2.y)^2)
end

function isANewNode(Nodes::Vector{Point}, New_Node::Point)
    push!(Nodes, New_Node)
end

function findTheClosest(sample::Point, Nodes::Vector{Point})
    min_dist = Inf
    closest_point = nothing
    for node in Nodes
        d = distance(node, sample)
        if d < min_dist
            min_dist = d
            closest_point = node
        end
    end
    return closest_point
end

function expandTowardsNearestNeighbor(start::Point, end::Point, step_size::Float64)
    dist = distance(start, end)
    if dist <= step_size
        return end
    else
        dx = step_size * (end.x - start.x) / dist
        dy = step_size * (end.y - start.y) / dist
        return Point(start.x + dx, start.y + dy)
    end
end


mutable struct RRT
    nodes::Vector{Point}
    params::PlannerParams
    scene::Scene

    function RRT(start::Point, scene::Scene, params::PlannerParams)
        return new([start], params, scene)
    end
end

function plan_path(start::Point, dest::Point, scene::Scene, params::PlannerParams)
    rrt = RRT(start, scene, params)
    iter = 0
    while iter < params.rrt_iters
        sample = sampleAPoint(scene)
        if isInObstacle(scene, sample)
            continue
        end
        closest_node = findTheClosest(sample, rrt.nodes)
        new_node = expandTowardsNearestNeighbor(closest_node, sample, params.rrt_dt)
        isANewNode(rrt.nodes, new_node)
        if distance(new_node, dest) < params.rrt_dt
            isANewNode(rrt.nodes, dest)
            break
        end
        iter += 1
    end
    return rrt.nodes
end
