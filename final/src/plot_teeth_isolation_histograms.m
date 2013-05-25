function plot_teeth_isolation_histograms(data_file, image_file)
  % load histogram data
  load(data_file);

  % create figure, plot histogram data and save to requested file
  fh = figure;
  hold on
  
  % two slices, one for upper jaw and one for lower jar
  % upper
  top = max(upper.histogram) * 1.1
  len = length(upper.histogram)
  subplot(2, 1, 1)
  plot(upper.histogram, 'lineWidth', 4);
  hold on;
  for split = 1:5
    plot([upper.splits(split),upper.splits(split)],[0,top],'r', 'lineWidth', 4)
  endfor
  legend('summed intensities', 'splits')
  axis([0 len 0 top])

  % lower
  top = max(lower.histogram) * 1.1
  len = length(lower.histogram)
  subplot(2, 1, 2)
  plot(lower.histogram, 'lineWidth', 4);
  hold on;
  for split = 1:5
    plot([lower.splits(split),lower.splits(split)],[0,top],'r', 'lineWidth', 4)
  endfor
  legend('summed intensities', 'splits')
  axis([0 len 0 top])


  print(image_file, '-tight', '-color');
  close(fh);
end
