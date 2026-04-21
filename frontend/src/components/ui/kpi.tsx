import { Card, CardContent } from './card';

export function KPI({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: any;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-slate-100">
            <Icon className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="text-xs text-slate-500">{label}</div>
            <div className="text-xl font-semibold">{value}</div>
            <div className="text-xs text-slate-500">{sub}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

