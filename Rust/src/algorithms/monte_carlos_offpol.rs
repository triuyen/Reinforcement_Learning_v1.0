use crate::contracts::model_free_env::ModelFreeEnv;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;


// Monte Carlo Agent struct
pub struct MonteCarloAgent<E: ModelFreeEnv> {
    pub env: E,
    pub episodes: usize,
    pub gamma: f32,
    pub q_value: Vec<Vec<f32>>,     // Using Vec<Vec<f32>> instead of HashMap
    pub returns: Vec<Vec<Vec<f32>>>, // Using Vec<Vec<Vec<f32>>>
}

impl<E: ModelFreeEnv> MonteCarloAgent<E> {

    // Constructor for the Monte Carlo Agent //
    pub fn new(mut env: E, episodes: usize, gamma: f32) -> Self {
        
        let num_states = E::num_states();
        let num_actions = E::num_actions(); 

        // Initialize q_value as a 2D vector of size [num_states][num_actions] with initial values of 0.0
        let q_value = vec![vec![0.0; num_actions]; num_states];

        // Initialize returns as a 3D vector: [num_states][num_actions] with empty Vec<f32> for each entry
        let returns = vec![vec![Vec::new(); num_actions]; num_states];
        
        env.reset();

        Self {
            env,
            episodes,
            gamma,
            q_value,
            returns
        }
    }

   
    // Generate an episode
    pub fn generate_episode(&mut self) -> Vec<(usize, usize, f32)> {
        let mut episode = Vec::new();
        self.env.reset();
    
        while !self.env.is_game_over() {
            let state = self.env.state_id();
            self.resize_if_needed(state); // Ensure correct size

            self.env.step(action);
    
            let reward = self.env.score();
            episode.push((state, action, reward));
    
            // Check if the environment has changed dynamically after taking the action
            self.resize_if_needed(self.env.state_id());
        }
    
        episode
    }
    
    // Update q-values based on the episode
    pub fn update_q_values(&mut self, episode: Vec<(usize, usize, f32)>) {
        let mut goal = 0.0;
        let mut visited_state_action_pairs = HashSet::new();

        for (state, action, reward) in episode.iter().rev() {
            self.resize_if_needed(*state); // Ensure correct size

            goal = self.gamma * goal + reward;

            if !visited_state_action_pairs.contains(&(*state, *action)) {
                self.returns[*state][*action].push(goal);

                let avg = self.returns[*state][*action].iter().sum::<f32>() 
                          / self.returns[*state][*action].len() as f32;
                self.q_value[*state][*action] = avg;
                visited_state_action_pairs.insert((*state, *action));
            }
        }
    }

    // Method to dynamically resize q_value and returns if the number of actions changes
    fn resize_if_needed(&mut self, state: usize) {

        let num_actions = E::num_actions();

        // Ensure q_value and returns have enough states
        if self.q_value.len() <= state {
            println!(
                "Resizing q_value to accommodate new state {}: old_size={}, new_size={}",
                state,
                self.q_value.len(),
                state + 1
            );
            self.q_value.resize(state + 1, vec![]);
        }

        if self.returns.len() <= state {
            println!(
                "Resizing returns to accommodate new state {}: old_size={}, new_size={}",
                state,
                self.returns.len(),
                state + 1
            );
            self.returns.resize(state + 1, vec![]);
        }

        // Resize the action space for the given state
        if self.q_value[state].len() != num_actions {
            println!(
                "Resizing q_value for state {}: old_num_actions={}, new_num_actions={}",
                state,
                self.q_value[state].len(),
                num_actions );

            // Trim excess actions if the number of actions has decreased
            if self.q_value[state].len() > num_actions {
                self.q_value[state].truncate(num_actions);
            } else {
                self.q_value[state].resize(num_actions, 0.0);
            }
        }

        if self.returns[state].len() != num_actions {
            println!(
                "Resizing returns for state {}: old_num_actions={}, new_num_actions={}",
                state,
                self.returns[state].len(),
                num_actions
            );

            if self.returns[state].len() > num_actions {
                self.returns[state].truncate(num_actions);
            } else {
                self.returns[state].resize_with(num_actions, Vec::new);
            }
        }
    }
        

    // Train the agent
    pub fn train(&mut self) {
        for _ in 0..self.episodes {
            let episode = self.generate_episode();
            self.update_q_values(episode);
        }
    }
}
