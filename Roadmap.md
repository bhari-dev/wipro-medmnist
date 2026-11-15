## Project: Medical Image Classification - Neural Network Explainability (Roadmap)

### Milestones

| Date    | Milestone         |
|---------|-------------------|
| Jun 19  | Kickoff meeting   |
| Jun 26  | Dataset ready     |
| Jul 3   | Model frozen      |
| Jul 10  | XAI complete      |
| Jul 17  | Presentation      |

---

### Phase 1 — Setup & dataset (Jun 19 – Jun 26)
- Create GitLab repo, define issues and assign roles
- Decide on dataset (PneumoniaMNIST or PathMNIST)
- Download data
- Divide workload between all members

### Phase 2 — Model training (Jun 26 – Jul 3)
- Implement CNN in Flux.jl
- Train and evaluate on test set
- Write basic exploration/visualisation
- Write unit tests for core methods
- Freeze weights once target accuracy is reached (Stop training the model, lock it in place and don't update further)

### Phase 3 — Explainability (Jul 3 – Jul 10)
- Implement saliency maps
- Extract activations from second-to-last layer
- Visualise latent space with PCA or t-SNE
- Optionally explore Grad-CAM or Occlusion Sensitivity

### Phase 4 — Wrap-up (Jul 10 – Jul 17)
- Write docstrings and finalise README
- Complete unit tests
- Prepare and rehearse 4-minute presentation
