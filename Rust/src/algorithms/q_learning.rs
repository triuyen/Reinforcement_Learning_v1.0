use rand::prelude::SliceRandom;
use rand::Rng;
use crate::contracts::model_free_env::ModelFreeEnv;

pub fn q_learning<TEnv: ModelFreeEnv>(
    num_episodes: usize,
    learning_rate: f32,
    gamma: f32,
    mut epsilon: f32,
    env_name: &str,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    use std::fs;
    use std::io::Write;

    // Définir les chemins de sortie
    let dir_path = format!("data/{}/q_learning", env_name);
    let avg_rewards_path = format!("{}/{}_q_learning_averaged_rewards.csv", dir_path, env_name);
    let episode_rewards_path = format!("{}/{}_q_learning_episode_rewards.csv", dir_path, env_name);
    let cumulative_rewards_path = format!("{}/{}_q_learning_cumulative_rewards.csv", dir_path, env_name);
    fs::create_dir_all(&dir_path).unwrap();

    let mut avg_rewards_file = fs::File::create(avg_rewards_path).unwrap();
    let mut episode_rewards_file = fs::File::create(episode_rewards_path).unwrap();
    let mut cumulative_rewards_file = fs::File::create(cumulative_rewards_path).unwrap();

    writeln!(avg_rewards_file, "interval,average_reward").unwrap();
    writeln!(episode_rewards_file, "episode,reward").unwrap();
    writeln!(cumulative_rewards_file, "episode,cumulative_reward").unwrap();

    let mut q_values = vec![vec![0.0; TEnv::num_actions()]; TEnv::num_states()];
    let mut rng = rand::thread_rng();

    let epsilon_start = epsilon;
    let epsilon_end = 0.01;
    let decay_episodes = num_episodes / 2;

    let mut total_reward_batch = 0.0;
    let mut cumulative_reward = 0.0;

    for episode in 0..num_episodes {
        let mut env = TEnv::new();
        let mut total_reward = 0.0;

        epsilon = decay_epsilon(epsilon_start, epsilon_end, episode, decay_episodes);

        env.reset();
        while !env.is_game_over() {
            let state = env.state_id();
            let available_actions = env.available_actions();

            let action = if rng.gen::<f32>() < epsilon {
                *available_actions.choose(&mut rng).unwrap()
            } else {
                q_values[state]
                    .iter()
                    .enumerate()
                    .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
                    .unwrap()
                    .0
            };

            let previous_score = env.score();
            env.step(action);
            let reward = env.score() - previous_score;
            total_reward += reward;

            let next_state = env.state_id();
            let max_q_next = q_values[next_state]
                .iter()
                .max_by(|q1, q2| q1.partial_cmp(q2).unwrap())
                .unwrap();
            q_values[state][action] +=
                learning_rate * (reward + gamma * max_q_next - q_values[state][action]);
        }

        cumulative_reward += total_reward;

        writeln!(episode_rewards_file, "{},{}", episode, total_reward).unwrap();
        writeln!(cumulative_rewards_file, "{},{}", episode, cumulative_reward).unwrap();

        total_reward_batch += total_reward;

        if (episode + 1) % 10_000 == 0 {
            let average_reward = total_reward_batch / 10_000.0;
            writeln!(avg_rewards_file, "{},{}", (episode + 1) / 10_000, average_reward).unwrap();
            total_reward_batch = 0.0;
        }
    }

    let mut policy = vec![0usize; TEnv::num_states()];
    for s in 0..TEnv::num_states() {
        policy[s] = q_values[s]
            .iter()
            .enumerate()
            .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
            .unwrap()
            .0;
    }

    (q_values, policy)
}

fn decay_epsilon(start: f32, end: f32, episode: usize, max_episodes: usize) -> f32 {
    if episode >= max_episodes {
        end
    } else {
        start + (end - start) * (episode as f32 / max_episodes as f32)
    }
}
