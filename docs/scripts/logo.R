sysfonts::font_add_google("Zilla Slab", "pf", regular.wt = 500)

hexSticker::sticker(
  subplot = ~ plot.new(),
  s_x = 1,
  s_y = 1,
  s_width = 0.1,
  s_height = 0.1,
  package = "tinytopics",
  p_x = 1,
  p_y = 1,
  p_size = 26,
  h_size = 1.2,
  p_family = "pf",
  p_color = "#3F8CCC",
  h_fill = "#FFF9F2",
  h_color = "#3F8CCC",
  dpi = 320,
  filename = "docs/assets/logo.png"
)
