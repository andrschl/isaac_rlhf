def compute_success_metric(self, env_ids):
    # the longer it stays up, the better
    return self.episode_length_buf[env_ids].float().mean() / self.max_episode_length
