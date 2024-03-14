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
      { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: 'Examples',
        items: [
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/bkutasi/EEG-ML' }
    ]
  }
})
