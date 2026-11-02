# Standardised zooplankton survey index (GAM) vs SOGEM-LTL Ecospace
# --------------------------------------------------------------------
# Observed index: log10 biomass ~ Year (factor) + s(day of season) + s(Station, re)          [M1]
#                 + log10(tow depth range) + day/night + net type                            [M2]
# Annual version: cyclic s(day of year) over all seasons.
# Year effects are predicted at a standard date (season median) with station effects excluded
# and covariates at reference values -> index for a standard date and place, with SE.
# Model side: (a) the same M1 GAM fitted to tow-matched model values;
#             (b) model means over the season at sampled cells, from the NetCDF (no sampling artefacts).
# Core-station check: GEO1, CPF1, CPF2 only.
# Inputs are written by export_model_window_index.py (same folder).

suppressPackageStartupMessages({library(mgcv); library(dplyr); library(ggplot2); library(tidyr)})
here <- if (interactive()) getwd() else dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
tows <- read.csv(file.path(here, "gam_input_tows.csv"), check.names = FALSE)
win  <- read.csv(file.path(here, "gam_model_window_index.csv"))
figs <- normalizePath(file.path(here, "..", "..", "..", "eval", "figs"))
tabs <- normalizePath(file.path(here, "..", "..", "..", "eval", "tables"))

OFF <- 1e-6
groups  <- c("Total", "ZC1-EUP", "ZC2-AMP", "ZC3-DEC", "ZC4-CLG", "ZC5-CSM")
seasons <- c("Winter", "Spring", "Summer", "Fall", "Annual")
core    <- c("GEO1", "CPF1", "CPF2")

tows <- tows %>% mutate(Station = factor(Station), fYear = factor(Year), net = factor(net),
                        lrng = log10(towrng))

r_ci <- function(x, y, w = NULL) {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]; n <- length(x)
  if (n < 5) return(c(r = NA, lo = NA, hi = NA, p = NA, n = n))
  if (is.null(w)) { r <- cor(x, y) } else {
    w <- w[ok] / sum(w[ok]); mx <- sum(w * x); my <- sum(w * y)
    r <- sum(w * (x - mx) * (y - my)) / sqrt(sum(w * (x - mx)^2) * sum(w * (y - my)^2))
  }
  z <- atanh(r); se <- 1 / sqrt(n - 3)
  c(r = r, lo = tanh(z - 1.96 * se), hi = tanh(z + 1.96 * se), p = 2 * pnorm(-abs(z) / se), n = n)
}

fit_index <- function(d, yvar, covs = FALSE, annual = FALSE, station_re = TRUE) {
  d$y <- log10(d[[yvar]] + OFF)
  d <- d[is.finite(d$y), ]
  d <- droplevels(d)
  if (annual) {
    sm <- "s(doy, bs = 'cc', k = 10)"; knots <- list(doy = c(0.5, 366.5))
  } else {
    k <- max(3, min(5, length(unique(d$dos)) - 1)); sm <- sprintf("s(dos, k = %d)", k); knots <- NULL
  }
  st <- if (station_re && nlevels(d$Station) > 3) "+ s(Station, bs = 're')" else if (nlevels(d$Station) > 1) "+ Station" else ""
  cv <- if (covs) paste("+ lrng + night", if (nlevels(d$net) > 1) "+ net" else "") else ""
  f <- as.formula(paste("y ~ fYear +", sm, st, cv))
  g <- gam(f, data = d, method = "REML", knots = knots)
  nd <- data.frame(fYear = levels(d$fYear), dos = median(d$dos), doy = 182,
                   Station = levels(d$Station)[1], lrng = median(d$lrng), night = 0,
                   net = factor(names(which.max(table(d$net))), levels = levels(d$net)))
  ex <- if (grepl("bs = 're'", st)) "s(Station)" else NULL
  if (!is.null(ex)) { p <- predict(g, nd, se.fit = TRUE, exclude = ex) } else { p <- predict(g, nd, se.fit = TRUE) }
  data.frame(Year = as.integer(levels(d$fYear)), index = as.numeric(p$fit), se = as.numeric(p$se.fit),
             n_tows = as.integer(table(d$fYear)[levels(d$fYear)]), dev_expl = summary(g)$dev.expl)
}

idx_all <- list(); res <- list()
for (grp in groups) for (s in seasons) {
  d <- if (s == "Annual") tows else filter(tows, Season == s)
  d <- d[!is.na(d[[grp]]) & !is.na(d[[paste0("EWE-", grp)]]), ]
  ann <- s == "Annual"
  o1 <- fit_index(d, grp, covs = FALSE, annual = ann)
  o2 <- fit_index(d, grp, covs = TRUE,  annual = ann)
  m1 <- fit_index(d, paste0("EWE-", grp), covs = FALSE, annual = ann)
  dc <- filter(d, Station %in% core)
  oc <- if (n_distinct(dc$Year) >= 6) fit_index(dc, grp, annual = ann, station_re = FALSE) else NULL
  mc <- if (!is.null(oc)) fit_index(dc, paste0("EWE-", grp), annual = ann, station_re = FALSE) else NULL
  mw <- filter(win, group == grp, season == s) %>% select(Year, log10_model)

  j <- o2 %>% select(Year, o2 = index, se2 = se) %>%
    left_join(select(o1, Year, o1 = index), by = "Year") %>%
    left_join(select(m1, Year, m1 = index), by = "Year") %>%
    left_join(mw, by = "Year")
  idx_all[[paste(grp, s)]] <- j %>% mutate(group = grp, season = s)

  add <- function(lbl, x, y, w = NULL) {
    r <- r_ci(x, y, w); data.frame(group = grp, season = s, comparison = lbl, r = r["r"], lo = r["lo"], hi = r["hi"],
                                   p = r["p"], n_years = r["n"], row.names = NULL)
  }
  out <- rbind(
    add("obs M1 (DOY+station) vs model window", j$o1, j$log10_model),
    add("obs M2 (+towrange, night, net) vs model window", j$o2, j$log10_model),
    add("obs M2 vs model window, weighted 1/SE^2", j$o2, j$log10_model, w = 1 / j$se2^2),
    add("obs M1 vs model M1 (same GAM, tow-matched)", j$o1, j$m1))
  if (!is.null(oc)) {
    jc <- oc %>% select(Year, oc = index) %>% left_join(select(mc, Year, mc = index), by = "Year") %>% left_join(mw, by = "Year")
    out <- rbind(out, add("core stations obs vs model window", jc$oc, jc$log10_model),
                      add("core stations obs vs model (same GAM, tow-matched)", jc$oc, jc$mc))
  }
  res[[paste(grp, s)]] <- out
}
res <- bind_rows(res); idx <- bind_rows(idx_all)
write.csv(res, file.path(tabs, "ecospace_zoop_gam_index_comparison_SC300.csv"), row.names = FALSE)
write.csv(idx, file.path(tabs, "ecospace_zoop_gam_index_values_SC300.csv"), row.names = FALSE)

cat("\nTOTAL CRUSTACEANS: r [95% CI] (n years)\n")
tot <- filter(res, group == "Total") %>% mutate(v = sprintf("%+.2f [%+.2f,%+.2f] n=%d", r, lo, hi, n_years))
print(as.data.frame(pivot_wider(select(tot, comparison, season, v), names_from = season, values_from = v)), right = FALSE)
cat("\nBY GROUP, obs M2 vs model window (weighted):\n")
g2 <- filter(res, comparison == "obs M2 vs model window, weighted 1/SE^2") %>%
  mutate(v = sprintf("%+.2f%s", r, ifelse(!is.na(p) & p < 0.05, "*", "")))
print(as.data.frame(pivot_wider(select(g2, group, season, v), names_from = season, values_from = v)))

# figure: Total, z-scored indices, obs with 95% CI
pz <- idx %>% filter(group == "Total") %>% group_by(season) %>%
  mutate(sd_o = sd(o2, na.rm = TRUE), obs_z = (o2 - mean(o2, na.rm = TRUE)) / sd_o, obs_lo = obs_z - 1.96 * se2 / sd_o,
         obs_hi = obs_z + 1.96 * se2 / sd_o, mod_z = (log10_model - mean(log10_model, na.rm = TRUE)) / sd(log10_model, na.rm = TRUE)) %>%
  ungroup() %>% mutate(season = factor(season, levels = seasons))
lab <- tot %>% filter(comparison == "obs M2 vs model window, weighted 1/SE^2") %>%
  mutate(season = factor(season, levels = seasons), txt = sprintf("r = %.2f [%.2f, %.2f]", r, lo, hi))
p <- ggplot(pz, aes(Year)) +
  geom_hline(yintercept = 0, colour = "grey70") +
  geom_errorbar(aes(ymin = obs_lo, ymax = obs_hi), colour = "darkorange", width = 0, alpha = 0.6) +
  geom_point(aes(y = obs_z, shape = "Observed (standardised survey index, 95% CI)"), colour = "darkorange", size = 2) +
  geom_line(aes(y = mod_z), colour = "blue") +
  geom_point(aes(y = mod_z, shape = "SOGEM-LTL (seasonal mean at sampled cells)"), colour = "blue", size = 1.8) +
  geom_text(data = lab, aes(x = 2000, y = Inf, label = txt), hjust = 0, vjust = 1.4, size = 3, inherit.aes = FALSE) +
  facet_wrap(~season, ncol = 1, scales = "free_y") +
  scale_shape_manual(values = c(16, 15), name = NULL) +
  labs(y = "Standardised index (z-score)", x = NULL,
       title = "Total crustacean zooplankton: GAM-standardised survey index vs SOGEM-LTL Ecospace") +
  theme_bw(base_size = 10) + theme(legend.position = "top")
ggsave(file.path(figs, "ecospace_SC300_zoop_gam_index_vs_model.png"), p, width = 7.5, height = 10, dpi = 300)
cat("\nsaved figure and tables\n")
