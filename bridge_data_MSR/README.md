BridgeData MSR dataset is identical in format to [BridgeData from UC Berkeley](https://rail-berkeley.github.io/bridgedata/) but differs from the latter in the following aspects:

- It was collected at Microsoft Research - Redmond. Otherwise, the environment is similar to BridgeData Berkeley's `toykitchen2`. The robot arm is a WidowX 250 6DOF. The scenes' background is generally more cluttered than in BridgeData Berkeley.

- All episodes have 3 camera views. This dataset doesn't have a wrist camera view.

- Episode lengths are up to `60` time steps, with an important exception described below.

- A large fraction of the dataset consists of so-called **play data**, i.e., trajectories where the robot performs a series of object manipualtions (picking, placing, pushing, etc) without a clear goal or clear time boundaries where one manipulation ends and the next one begins. The play trajectories can be up to `160` time steps long.

The dataset conversion code was adapted from [Karl Pertsch's Brdige dataset conversion repo](https://github.com/kpertsch/bridge_rlds_builder/tree/main).