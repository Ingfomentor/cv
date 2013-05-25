function plot_crown_histogram(data_file, crown, image_file)
  % load histogram data
  load(data_file);

  % convert to 1-based indexing
  crown = crown + 1

  % 4 upper and 4 lower teeth ...
  if crown < 5
    histogram = histograms_upper(crown,:);
    mu        = double(mus_upper(crown));
    sigma     = sigmas_upper(crown);
  else
    histogram = histograms_lower(crown-4,:);
    mu        = double(mus_lower(crown-4));
    sigma     = sigmas_lower(crown-4);
  endif
  
  % create figure, plot histogram data and save to requested file
  fh = figure;
  
  % histogram
  top = max(histogram) * 1.1;
  len = length(histogram);

  plot(histogram, 'lineWidth', 3);
  hold on;

  % plot Gaussian/Normal curve
  X = 0:len;
  Y = normpdf(X, mu, sigma);
  plot(Y, 'r', 'lineWidth', 4)

  legend('histogram', 'eerste Gauss component')
  axis([0 len 0 top])

  print(image_file, '-tight', '-color');
  close(fh);
end
