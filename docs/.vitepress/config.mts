import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/EEG-ML/",
  title: "EEG-ML",
  description: "EEG-ML is a proof of concept project for high precision classification of movement associated brainwaves.",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    search: {
			provider: "local",
		},
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Docs', link: '/getting-started/getting-started' }
    ],

    sidebar: [
      {
        text: 'Start  here',
        items: [
          { text: 'Getting started', link: '/getting-started/getting-started' },
          { text: 'Installation', link: '/getting-started/installation' },
          { text: 'Important changes', link: '/getting-started/important-changes'}
        ]
      },
      // Introduction
      {
        text: 'Introduction',
        items: [
          { text: 'Background', link: '/introduction/background' },
          { text: 'Project Goals', link: '/introduction/project-goals' }
        ]
      },
      // Methods
      {
        text: 'Methods',
        items: [
          { text: 'Data Collection', link: '/methods/data-collection' },
          { text: 'Data Preprocessing', link: '/methods/data-preprocessing' },
          { text: 'Feature Extraction', link: '/methods/feature-extraction' },
          { text: 'Model Training', link: '/methods/model-training' },
          { text: 'Model Evaluation', link: '/methods/model-evaluation' }
        ]
      },
      // Results
      {
        text: 'Results',
        items: [
          { text: 'Dataset creation', link: '/results/dataset-creation' },
          { text: 'Data Analysis', link: '/results/data-analysis' },
          { text: 'Model Performance', link: '/results/model-performance' },
        ]
      },
      // Summary
      {
        text: 'Summary',
        items: [
          { text: 'Conclusions', link: '/summary/conclusions' },
          { text: 'Future Work', link: '/summary/future-work' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/bkutasi/EEG-ML' }
    ]
  }
})
